#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define eps 2.2204460492503131e-16

// Importing the exact CalculateDistance function from msfm2d - for mex.c
extern double CalculateDistance(double *T, double Fij, int *dims, int i, int j, bool usesecond, bool usecross, bool *Frozen) {
    /* Derivatives */
    double Tm[4]={0, 0, 0, 0};
    double Tm2[4]={0, 0, 0, 0};
    double Coeff[3];
    
    /* local derivatives in distance image */
    double Tpatch_2_3, Txm2, Tpatch_4_3, Txp2;
    double Tpatch_3_2, Tym2, Tpatch_3_4, Typ2;
    double Tpatch_2_2, Tr1m2, Tpatch_4_4, Tr1p2;
    double Tpatch_2_4, Tr2m2, Tpatch_4_2, Tr2p2;
    
    /* Return values root of polynomial */
    double ansroot[2]={0, 0};
    
    /* Loop variables  */
    int q, t;
    
    /* Derivative checks */
    bool ch1, ch2;
    
    /* Order derivatives in a certain direction */
    int Order[4]={0, 0, 0, 0};
    
    /* Current location */
    int in, jn;
    
    /* Constant cross term */
    const double c1=0.5;
    
    double Tt, Tt2;
    /*Get First order derivatives (only use frozen pixel)  */
    in=i-1; jn=j+0; if(isfrozen2d(in, jn, dims, Frozen)) { Tpatch_2_3=T[mindex2(in, jn, dims[0])]; } else { Tpatch_2_3=INF; }
    in=i+0; jn=j-1; if(isfrozen2d(in, jn, dims, Frozen)) { Tpatch_3_2=T[mindex2(in, jn, dims[0])]; } else { Tpatch_3_2=INF; }
    in=i+0; jn=j+1; if(isfrozen2d(in, jn, dims, Frozen)) { Tpatch_3_4=T[mindex2(in, jn, dims[0])]; } else { Tpatch_3_4=INF; }
    in=i+1; jn=j+0; if(isfrozen2d(in, jn, dims, Frozen)) { Tpatch_4_3=T[mindex2(in, jn, dims[0])]; } else { Tpatch_4_3=INF; }
    if(usecross) {
        in=i-1; jn=j-1; if(isfrozen2d(in, jn, dims, Frozen)) { Tpatch_2_2=T[mindex2(in, jn, dims[0])]; } else { Tpatch_2_2=INF; }
        in=i-1; jn=j+1; if(isfrozen2d(in, jn, dims, Frozen)) { Tpatch_2_4=T[mindex2(in, jn, dims[0])]; } else { Tpatch_2_4=INF; }
        in=i+1; jn=j-1; if(isfrozen2d(in, jn, dims, Frozen)) { Tpatch_4_2=T[mindex2(in, jn, dims[0])]; } else { Tpatch_4_2=INF; }
        in=i+1; jn=j+1; if(isfrozen2d(in, jn, dims, Frozen)) { Tpatch_4_4=T[mindex2(in, jn, dims[0])]; } else { Tpatch_4_4=INF; }
    }
    /*The values in order is 0 if no neighbours in that direction  */
    /*1 if 1e order derivatives is used and 2 if second order  */
    /*derivatives are used  */
    Order[0]=0; Order[1]=0; Order[2]=0; Order[3]=0;
    /*Make 1e order derivatives in x and y direction  */
    Tm[0] = min( Tpatch_2_3 , Tpatch_4_3); if(IsFinite(Tm[0])){ Order[0]=1; }
    Tm[1] = min( Tpatch_3_2 , Tpatch_3_4); if(IsFinite(Tm[1])){ Order[1]=1; }
    /*Make 1e order derivatives in cross directions  */
    if(usecross) {
        Tm[2] = min( Tpatch_2_2 , Tpatch_4_4); if(IsFinite(Tm[2])){ Order[2]=1; }
        Tm[3] = min( Tpatch_2_4 , Tpatch_4_2); if(IsFinite(Tm[3])){ Order[3]=1; }
    }
	
    /*Make 2e order derivatives  */
    if(usesecond) {
        /*Get Second order derivatives (only use frozen pixel) */
        in=i-2; jn=j+0; if(isfrozen2d(in, jn, dims, Frozen)) { Txm2=T[mindex2(in, jn, dims[0])]; } else { Txm2=INF; }
        in=i+2; jn=j+0; if(isfrozen2d(in, jn, dims, Frozen)) { Txp2=T[mindex2(in, jn, dims[0])]; } else { Txp2=INF; }
        in=i+0; jn=j-2; if(isfrozen2d(in, jn, dims, Frozen)) { Tym2=T[mindex2(in, jn, dims[0])]; } else { Tym2=INF; }
        in=i+0; jn=j+2; if(isfrozen2d(in, jn, dims, Frozen)) { Typ2=T[mindex2(in, jn, dims[0])]; } else { Typ2=INF; }
        if(usecross) {
            in=i-2; jn=j-2; if(isfrozen2d(in, jn, dims, Frozen)) { Tr1m2=T[mindex2(in, jn, dims[0])]; } else { Tr1m2=INF; }
            in=i-2; jn=j+2; if(isfrozen2d(in, jn, dims, Frozen)) { Tr2m2=T[mindex2(in, jn, dims[0])]; } else { Tr2m2=INF; }
            in=i+2; jn=j-2; if(isfrozen2d(in, jn, dims, Frozen)) { Tr2p2=T[mindex2(in, jn, dims[0])]; } else { Tr2p2=INF; }
            in=i+2; jn=j+2; if(isfrozen2d(in, jn, dims, Frozen)) { Tr1p2=T[mindex2(in, jn, dims[0])]; } else { Tr1p2=INF; }
        }
        
        Tm2[0]=0; Tm2[1]=0;Tm2[2]=0; Tm2[3]=0;
        /*pixels with a pixeldistance 2 from the center must be */
        /*lower in value otherwise use other side or first order */
        ch1=(Txm2<Tpatch_2_3)&&IsFinite(Tpatch_2_3); ch2=(Txp2<Tpatch_4_3)&&IsFinite(Tpatch_4_3);
        if(ch1&&ch2) {
            Tm2[0] =min( (4.0*Tpatch_2_3-Txm2)/3.0 , (4.0*Tpatch_4_3-Txp2)/3.0);  Order[0]=2;
        }
        else if (ch1) {
            Tm2[0]=(4.0*Tpatch_2_3-Txm2)/3.0; Order[0]=2;
        }
        else if(ch2) {
            Tm2[0] =(4.0*Tpatch_4_3-Txp2)/3.0; Order[0]=2;
        }
        
        ch1=(Tym2<Tpatch_3_2)&&IsFinite(Tpatch_3_2); ch2=(Typ2<Tpatch_3_4)&&IsFinite(Tpatch_3_4);
        
        if(ch1&&ch2) {
            Tm2[1] =min( (4.0*Tpatch_3_2-Tym2)/3.0 , (4.0*Tpatch_3_4-Typ2)/3.0); Order[1]=2;
        }
        else if(ch1) {
            Tm2[1]=(4.0*Tpatch_3_2-Tym2)/3.0; Order[1]=2;
        }
        else if(ch2) {
            Tm2[1]=(4.0*Tpatch_3_4-Typ2)/3.0; Order[1]=2;
        }
        if(usecross) {
            ch1=(Tr1m2<Tpatch_2_2)&&IsFinite(Tpatch_2_2); ch2=(Tr1p2<Tpatch_4_4)&&IsFinite(Tpatch_4_4);
            if(ch1&&ch2) {
                Tm2[2] =min( (4.0*Tpatch_2_2-Tr1m2)/3.0 , (4.0*Tpatch_4_4-Tr1p2)/3.0); Order[2]=2;
            }
            else if(ch1) {
                Tm2[2]=(4.0*Tpatch_2_2-Tr1m2)/3.0; Order[2]=2;
            }
            else if(ch2){
                Tm2[2]=(4.0*Tpatch_4_4-Tr1p2)/3.0; Order[2]=2;
            }
            
            ch1=(Tr2m2<Tpatch_2_4)&&IsFinite(Tpatch_2_4); ch2=(Tr2p2<Tpatch_4_2)&&IsFinite(Tpatch_4_2);
            if(ch1&&ch2){
                Tm2[3] =min( (4.0*Tpatch_2_4-Tr2m2)/3.0 , (4.0*Tpatch_4_2-Tr2p2)/3.0); Order[3]=2;
            }
            else if(ch1) {
                Tm2[3]=(4.0*Tpatch_2_4-Tr2m2)/3.0; Order[3]=2;
            }
            else if(ch2) {
                Tm2[3]=(4.0*Tpatch_4_2-Tr2p2)/3.0; Order[3]=2;
            }
        }
    }
    /*Calculate the distance using x and y direction */
    Coeff[0]=0; Coeff[1]=0; Coeff[2]=-1/(max(pow2(Fij),eps));
    
    for (t=0; t<2; t++) {
        switch(Order[t]) {
            case 1:
                Coeff[0]+=1; Coeff[1]+=-2*Tm[t]; Coeff[2]+=pow2(Tm[t]);
                break;
            case 2:
                Coeff[0]+=(2.2500); Coeff[1]+=-2.0*Tm2[t]*(2.2500); Coeff[2]+=pow2(Tm2[t])*(2.2500);
                break;
        }
    }
    roots(Coeff, ansroot);
    Tt=max(ansroot[0], ansroot[1]);
    /*Calculate the distance using the cross directions */
    if(usecross) {
        /* Original Equation */
        /*    Coeff[0]=0; Coeff[1]=0; Coeff[2]=-1/(max(pow2(Fij),eps)) */
        Coeff[0]+=0; Coeff[1]+=0; Coeff[2]+=-1/(max(pow2(Fij),eps));
        for (t=2; t<4; t++) {
            switch(Order[t]) {
                case 1:
                    Coeff[0]+=c1; Coeff[1]+=-2.0*c1*Tm[t]; Coeff[2]+=c1*pow2(Tm[t]);
                    break;
                case 2:
                    Coeff[0]+=c1*2.25; Coeff[1]+=-2*c1*Tm2[t]*(2.25); Coeff[2]+=pow2(Tm2[t])*c1*2.25;
                    break;
            }
        }
        if(Coeff[0]>0) {
            roots(Coeff, ansroot);
            Tt2=max(ansroot[0], ansroot[1]);
            /*Select minimum distance value of both stensils */
            Tt=min(Tt, Tt2);
        }
    }
    /*Upwind condition check, current distance must be larger */
    /*then direct neighbours used in solution */
    /*(Will this ever happen?) */
    if(usecross) {
        for(q=0; q<4; q++) { 
            if(IsFinite(Tm[q])&&(Tt<Tm[q])) 
            { 
                Tt=Tm[minarray(Tm, 4)]+(1/(max(Fij,eps)));
            }
        }
    }
    else {
        for(q=0; q<2; q++)
        { 
            if(IsFinite(Tm[q])&&(Tt<Tm[q])) {
                Tt=Tm[minarray(Tm, 2)]+(1/(max(Fij,eps)));}
        }
    }
    return Tt;
}

// Structure for heap nodes
typedef struct {
    int index;
    double value;
} HeapNode;

// Min-heap structure
typedef struct {
    HeapNode *nodes;
    int *position;
    int size;
    int capacity;
} MinHeap;

// Function to swap two heap nodes
void swap(HeapNode *a, HeapNode *b) {
    HeapNode temp = *a;
    *a = *b;
    *b = temp;
}

// Heapify function for min-heap
void heapify(MinHeap *heap, int i) {
    int smallest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < heap->size && heap->nodes[left].value < heap->nodes[smallest].value)
        smallest = left;
    if (right < heap->size && heap->nodes[right].value < heap->nodes[smallest].value)
        smallest = right;

    if (smallest != i) {
        swap(&heap->nodes[i], &heap->nodes[smallest]);
        heap->position[heap->nodes[i].index] = i;
        heap->position[heap->nodes[smallest].index] = smallest;
        heapify(heap, smallest);
    }
}

HeapNode extractMin(MinHeap *heap) {
    HeapNode root = heap->nodes[0];
    heap->nodes[0] = heap->nodes[heap->size - 1];
    heap->position[heap->nodes[0].index] = 0;
    heap->size--;
    heapify(heap, 0);
    return root;
}

void insertHeap(MinHeap *heap, int index, double value) {
    int i = heap->size;
    heap->nodes[i].index = index;
    heap->nodes[i].value = value;
    heap->position[index] = i;
    heap->size++;
    while (i > 0 && heap->nodes[(i - 1) / 2].value > heap->nodes[i].value) {
        swap(&heap->nodes[i], &heap->nodes[(i - 1) / 2]);
        heap->position[heap->nodes[i].index] = i;
        heap->position[heap->nodes[(i - 1) / 2].index] = (i - 1) / 2;
        i = (i - 1) / 2;
    }
}

MinHeap *createMinHeap(int capacity) {
    MinHeap *heap = (MinHeap *)malloc(sizeof(MinHeap));
    heap->nodes = (HeapNode *)malloc(capacity * sizeof(HeapNode));
    heap->position = (int *)malloc(capacity * sizeof(int));
    heap->capacity = capacity;
    heap->size = 0;
    for (int i = 0; i < capacity; i++)
        heap->position[i] = -1;
    return heap;
}

void freeHeap(MinHeap *heap) {
    free(heap->nodes);
    free(heap->position);
    free(heap);
}

void initialize_T(double *T, bool *Frozen, int size) {
    for (int i = 0; i < size; i++) {
        T[i] = INF;
        Frozen[i] = false;
    }
}

void msfm2d(double *T, const double *F, const int *source_points, int num_sources, int rows, int cols, bool use_second, bool use_cross) {
    int size = rows * cols;
    bool *Frozen = (bool *)malloc(size * sizeof(bool));
    MinHeap *heap = createMinHeap(size);

    initialize_T(T, Frozen, size);
    
    //printf("Initializing source points...\n");
    for (int s = 0; s < num_sources; s++) {
        int i = source_points[2 * s];
        int j = source_points[2 * s + 1];
        int idx = i + j * rows;
        //printf("Inserting source point at (%d, %d) -> idx = %d\n", i, j, idx);
        T[idx] = 0.0;
        insertHeap(heap, idx, 0.0);
        //printf("Heap size after insertion: %d\n", heap->size);
    }

    while (heap->size > 0) {
        HeapNode minNode = extractMin(heap);
        int idx = minNode.index;
        int i = idx % rows;
        int j = idx / rows;
        //printf("Processing node (%d, %d) -> idx = %d with value: %f\n", i, j, idx, minNode.value);
        Frozen[idx] = true;
        int neighbors[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int n = 0; n < 4; n++) {
            int ni = i + neighbors[n][0];
            int nj = j + neighbors[n][1];
            if (ni < 0 || ni >= rows || nj < 0 || nj >= cols) continue;
            int neighbor_idx = ni + nj * rows;
            if (Frozen[neighbor_idx]) continue;
            //printf("Checking neighbor (%d, %d) -> idx %d\n", ni, nj, neighbor_idx);
            double new_T = CalculateDistance(T, F[neighbor_idx], (int[]){rows, cols}, ni, nj, use_second, use_cross, Frozen);
            if (new_T < T[neighbor_idx]) {
                T[neighbor_idx] = new_T;
                insertHeap(heap, neighbor_idx, new_T);
            }
            //printf("T[neighbor] after update: %f\n", T[neighbor_idx]);
        }
    }
    free(Frozen);
    freeHeap(heap);
}
