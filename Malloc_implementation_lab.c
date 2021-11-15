/*
 * CS 208 Lab 4: Malloc Lab
 *
 * <Walt Li, Eric Gassel>
 *
 * Simple allocator based on explicit free lists, first fit search,
 * and boundary tag coalescing.
 *
 * Each block has header and footer of the form:
 *
 *      63                  4  3  2  1  0
 *      ------------------------------------
 *     | s  s  s  s  ... s  s  0  0  0  a/f |
 *      ------------------------------------
 *
 * where s are the meaningful size bits and a/f is 1
 * if and only if the block is allocated. The list has the following form:
 *
 * begin                                                             end
 * heap                                                             heap
 *  -------------------------------------------------------------------------------
 * |  pointer to head  | hdr(16:a) | ftr(16:a) | zero or more usr blks | hdr(0:a) |
 *  -------------------------------------------------------------------------------
 * |                   |       prologue        |                       | epilogue |
 * |                   |         block         |                       | block    |
 *
 * The allocated prologue and epilogue blocks are overhead that
 * eliminate edge conditions during coalescing.
 * 
 * Each free block has the following structure
 * 
 * begin                                                             end
 * block                                                            block
 *  ---------------------------------------------------------------------
 * |  hdr(size:f) |  prev address  |  next address  | ... |  ftr(size:f) |
 *  ---------------------------------------------------------------------
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>

#include "mm.h"
#include "memlib.h"

/* Basic constants and macros */
#define WSIZE       8       /* word size (bytes) */
#define DSIZE       16      /* doubleword size (bytes) */
#define CHUNKSIZE  (1<<12)  /* initial heap size (bytes) */
#define OVERHEAD    16      /* overhead of header and footer (bytes) */
#define MINIMUMSIZE 32      /* minimum size of a block (16 bytes overhead + 16 bytes payload) */

/* NOTE: feel free to replace these macros with helper functions and/or
 * add new ones that will be useful for you. Just make sure you think
 * carefully about why these work the way they do
 */

/* Pack a size and allocated bit into a word */
#define PACK(size, alloc)  ((size) | (alloc))

/* Read and write a word at address p */
#define GET(p)       (*(size_t *)(p))
#define PUT(p, val)  (*(size_t *)(p) = (val))

/* Perform unscaled pointer arithmetic */
#define PADD(p, val) ((char *)(p) + (val))
#define PSUB(p, val) ((char *)(p) - (val))

/* Read the size and allocated fields from address p */
#define GET_SIZE(p)  (GET(p) & ~0xf)
#define GET_ALLOC(p) (GET(p) & 0x1)

/* Given block ptr bp, compute address of its header and footer */
#define HDRP(bp)       (PSUB(bp, WSIZE))
#define FTRP(bp)       (PADD(bp, GET_SIZE(HDRP(bp)) - DSIZE))

/* Given block ptr bp, compute address of next and previous blocks */
#define NEXT_BLKP(bp)  (PADD(bp, GET_SIZE(HDRP(bp))))
#define PREV_BLKP(bp)  (PSUB(bp, GET_SIZE((PSUB(bp, DSIZE)))))

/* Pointer to the pointer of next and previous FREE blocks */
#define PREV_FREE(bp)  (*(char **)(bp))
#define NEXT_FREE(bp)  (*(char **)(bp + WSIZE))

/* Global variables */

/* 
 * Pointer to first byte in the heap, 
 * future used as the pointer to the address of linked-list head
 */
static void *heap_start;

/* Function prototypes for internal helper routines */

static bool check_heap(int lineno);
static void print_heap();
static void print_block(void *bp);
static bool check_block(int lineno, void *bp);
static void *extend_heap(size_t size);
static void *find_fit(size_t asize);
static void *coalesce(void *bp);
static void place(void *bp, size_t asize);
static size_t max(size_t x, size_t y);

/* Functions for linked-list manipulation */
static void insert_head(void *bp);
static void remove_node(void *bp);

/*
 * mm_init -- initiate the heap
 * return 0 if successfully initate heap, -1 if fail to expand heap
 */
int mm_init(void) {
    /* create the initial empty heap */
    if ((heap_start = mem_sbrk(4 * WSIZE)) == NULL) /* heap_start pointing to first byte in heap */
        return -1;

    PUT(heap_start, 0);                        /* alignment padding */
    PUT(PADD(heap_start, WSIZE), PACK(OVERHEAD, 1));  /* prologue header */
    PUT(PADD(heap_start, DSIZE), PACK(OVERHEAD, 1));  /* prologue footer */
    PUT(PADD(heap_start, WSIZE + DSIZE), PACK(0, 1));   /* epilogue header */

    /* Extend the empty heap with a free block of CHUNKSIZE bytes */
    if (extend_heap(CHUNKSIZE / WSIZE) == NULL)
        return -1;

    return 0;
}

/*
 * mm_malloc -- allocate the block if available free space is found, else extend heap, and allocate
 * argument: the size of the new block to allocate
 * return: the pointer to the newly allocated block, and NULL if size <= 0
 */
void *mm_malloc(size_t size) {
    size_t asize;      /* adjusted block size */
    size_t extendsize; /* amount to extend heap if no fit */
    char *bp;

    /* Ignore spurious requests */
    if (size <= 0)
        return NULL;

    /* Adjust block size to include overhead and alignment reqs. */
    if (size <= DSIZE) {
        asize = DSIZE + OVERHEAD;
    } else {
        /* Add overhead and then round up to nearest multiple of double-word alignment */
        asize = DSIZE * ((size + (OVERHEAD) + (DSIZE - 1)) / DSIZE);
    }

    /* Search the free list for a fit */
    if ((bp = find_fit(asize)) != NULL) {
        place(bp, asize);
        return bp;
    }

    /* No fit found. Get more memory and place the block */
    extendsize = max(asize, CHUNKSIZE);
    if ((bp = extend_heap(extendsize / WSIZE)) == NULL)
        return NULL;

    place(bp, asize);
    return bp;
}

/*
 * mm_free -- deallocate block
 * argument: pointer to the block to deallocate
 * return: nothing
 */
void mm_free(void *bp) {

    size_t size = GET_SIZE(HDRP(bp));  

    /* change flag 1->0 in header and footer */
    PUT(HDRP(bp), PACK(size, 0));
    PUT(FTRP(bp), PACK(size, 0));

    coalesce(bp);
}


/* The remaining routines are internal helper routines */


/*
 * place -- Place block of asize bytes at start of free block bp
 *          and make new free block overhead if free space over minimum block size
 * argument: a pointer to a free block and the size of block to place inside it
 * return: nothing
 * conditions: check whether free space after allocation is over minimum block size
 */
static void place(void *bp, size_t asize) {
    
    /* Calculate free space left */
    int free_space_left = GET_SIZE(HDRP(bp)) - asize;
    
    /* Check if there is enough space for the minimum block size */
    if (free_space_left < MINIMUMSIZE) {
        /* Free space useless, just include it in the present block */
        PUT(HDRP(bp), PACK(GET_SIZE(HDRP(bp)), 1));
        PUT(FTRP(bp), PACK(GET_SIZE(HDRP(bp)), 1));
        remove_node(bp);
    }
    else { 
        // update header and footer to allocated
        PUT(HDRP(bp), PACK(asize, 1));
        PUT(FTRP(bp), PACK(asize, 1));
        remove_node(bp);
        // create unallocated header following the footer of this block
        bp = NEXT_BLKP(bp);
        PUT(HDRP(bp), PACK(free_space_left, 0));
        PUT(FTRP(bp), PACK(free_space_left, 0));
        insert_head(bp);
    }
}

/*
 * coalesce -- Boundary tag coalescing.
 * Takes a pointer to a free block
 * Return ptr to coalesced block
 * Conditions: both adjacent blocks allocated, only previous/next block allocated, and both free
 */
static void *coalesce(void *bp) {

    size_t prev_alloc = GET_ALLOC(PSUB(bp, DSIZE));
    size_t next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));
    size_t size = GET_SIZE(HDRP(bp));
    
    /* both adjecent blocks are allocated, no need to do anything */
    if (prev_alloc && next_alloc) {
        insert_head(bp);
    }

    else if (prev_alloc && !next_alloc) { /* only next block is free */
        size += GET_SIZE(HDRP(NEXT_BLKP(bp)));
        /* remove the old node */
        remove_node(NEXT_BLKP(bp));
        PUT(HDRP(bp), PACK(size, 0));
        PUT(FTRP(bp), PACK(size,0));
        /* add the new node */
        insert_head(bp);
    }

    else if (!prev_alloc && next_alloc) { /* only prev block is free */
        size += GET_SIZE(HDRP(PREV_BLKP(bp)));
        bp = PREV_BLKP(bp);

        /* remove the old node */
        remove_node(bp);
        PUT(HDRP(bp), PACK(size, 0));
        PUT(FTRP(bp), PACK(size, 0));
        /* add new node */
        insert_head(bp);
    }

    else { /* both are free */
        size += GET_SIZE(HDRP(PREV_BLKP(bp))) + GET_SIZE(HDRP(NEXT_BLKP(bp)));

        /* remove the old nodes */
        remove_node(PREV_BLKP(bp));
        remove_node(NEXT_BLKP(bp));

        bp = PREV_BLKP(bp);
        PUT(HDRP(bp), PACK(size, 0));
        PUT(FTRP(bp), PACK(size, 0));
        /* add new node */
        insert_head(bp);
    }
    return bp;
}


/*
 * find_fit - Find a fit for a block with asize bytes
 */
static void *find_fit(size_t asize) {
    /* search from the start of the free linked-list to the end; the the last NEXT pointer points to the start of the heap */
    for (void *cur_block = heap_start; cur_block != mem_heap_lo(); cur_block = NEXT_FREE(cur_block)) {
        if (asize <= GET_SIZE(HDRP(cur_block))) {
            return cur_block;
        }
    } 
    return NULL;  /* no fit found */
}

/*
 * extend_heap - Extend heap with free block and return its block pointer
 */
static void *extend_heap(size_t words) {
    char *bp;
    size_t size;

    /* Allocate an even number of words to maintain alignment */
    size = words * WSIZE;
    if (words % 2 == 1)
        size += WSIZE;
    // printf("extending heap to %zu bytes\n", mem_heapsize());
    if ((long)(bp = mem_sbrk(size)) < 0)
        return NULL;

    /* Initialize free block header/footer and the epilogue header */
    PUT(HDRP(bp), PACK(size, 0));         /* free block header */
    PUT(FTRP(bp), PACK(size, 0));         /* free block footer */
    PUT(HDRP(NEXT_BLKP(bp)), PACK(0, 1)); /* new epilogue header */

    /* Coalesce if the previous block was free */
    return coalesce(bp);
}

/*
 * check_heap -- Performs basic heap consistency checks for an implicit free list allocator
 * and prints out all blocks in the heap in memory order.
 * Checks include proper prologue and epilogue, alignment, and matching header and footer
 * Takes a line number (to give the output an identifying tag).
 */
static bool check_heap(int line) {
    char *bp;

    if ((GET_SIZE(HDRP(heap_start)) != DSIZE) || !GET_ALLOC(HDRP(heap_start))) {
        printf("(check_heap at line %d) Error: bad prologue header\n", line);
        return false;
    }

    for (bp = heap_start; GET_SIZE(HDRP(bp)) > 0; bp = NEXT_BLKP(bp)) {
        if (!check_block(line, bp)) {
            return false;
        }
    }

    if ((GET_SIZE(HDRP(bp)) != 0) || !(GET_ALLOC(HDRP(bp)))) {
        printf("(check_heap at line %d) Error: bad epilogue header\n", line);
        return false;
    }

    return true;
}

/*
 * check_block -- Checks a block for alignment and matching header and footer
 */
static bool check_block(int line, void *bp) {
    if ((size_t)bp % DSIZE) {
        printf("(check_heap at line %d) Error: %p is not double-word aligned\n", line, bp);
        return false;
    }
    if (GET(HDRP(bp)) != GET(FTRP(bp))) {
        printf("(check_heap at line %d) Error: header does not match footer\n", line);
        return false;
    }
    return true;
}

/*
 * print_heap -- Prints out the current state of the implicit free list
 */
static void print_heap() {
    char *bp;

    printf("Heap (%p):\n", heap_start);

    for (bp = heap_start; GET_SIZE(HDRP(bp)) > 0; bp = NEXT_BLKP(bp)) {
        print_block(bp);
    }

    print_block(bp);
}

/*
 * print_block -- Prints out the current state of a block
 */
static void print_block(void *bp) {
    size_t hsize, halloc, fsize, falloc;

    hsize = GET_SIZE(HDRP(bp));
    halloc = GET_ALLOC(HDRP(bp));
    fsize = GET_SIZE(FTRP(bp));
    falloc = GET_ALLOC(FTRP(bp));

    if (hsize == 0) {
        printf("%p: End of free list\n", bp);
        return;
    }

    printf("%p: header: [%ld:%c] footer: [%ld:%c]\n", bp,
       hsize, (halloc ? 'a' : 'f'),
       fsize, (falloc ? 'a' : 'f'));
}

/*
 * max: returns x if x > y, and y otherwise.
 */
static size_t max(size_t x, size_t y) {
    return (x > y) ? x : y;
}

/*
 * insert_head: insert the current free block to the head of the linked list.
 */
static void insert_head(void *bp) {
    /* next pointer of the new node points to head */
    NEXT_FREE(bp) = heap_start;

    /* set the old head's prev pointer to new node */
    PREV_FREE(heap_start) = bp;

    /* set the prev pointer's next to NULL */
    PREV_FREE(bp) = NULL;

    /* set head to the new pointer */
    heap_start = bp;
}

/*
 * remove_node: remove the node from the linked list.
 */
static void remove_node(void *bp) {
    if (PREV_FREE(bp)) { /* if the target is NOT the head */
        /* update the next of the left to the right */
        NEXT_FREE(PREV_FREE(bp)) = NEXT_FREE(bp);
    }
    else { /* if the target node is the head */
        /* update head to the next */
        heap_start = NEXT_FREE(bp);
    }
    /* update the prev of the right to the left */
    PREV_FREE(NEXT_FREE(bp)) = PREV_FREE(bp);
}