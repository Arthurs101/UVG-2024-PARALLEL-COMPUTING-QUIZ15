#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <ctime>
#include <cmath>

#define TILE_WIDTH 16  // Define the width of tiles for shared memory optimization

// CUDA Kernel for matrix multiplication
__global__ void matrixMulKernel(float* d_A, float* d_B, float* d_C, int numAColumns, int numBColumns) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float Cvalue = 0;

    for (int tileIdx = 0; tileIdx < (numAColumns - 1) / TILE_WIDTH + 1; tileIdx++) {
        if (row < numAColumns && tileIdx * TILE_WIDTH + threadIdx.x < numAColumns)
            tile_A[threadIdx.y][threadIdx.x] = d_A[row * numAColumns + tileIdx * TILE_WIDTH + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0;

        if (tileIdx * TILE_WIDTH + threadIdx.y < numAColumns && col < numBColumns)
            tile_B[threadIdx.y][threadIdx.x] = d_B[(tileIdx * TILE_WIDTH + threadIdx.y) * numBColumns + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) {
            Cvalue += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < numAColumns && col < numBColumns) {
        d_C[row * numBColumns + col] = Cvalue;
    }
}

// Sequential matrix multiplication on CPU for verification
void matrixMulSequential(const float* A, const float* B, float* C, int numARows, int numAColumns, int numBColumns) {
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numBColumns; j++) {
            float sum = 0.0f;
            for (int k = 0; k < numAColumns; k++) {
                sum += A[i * numAColumns + k] * B[k * numBColumns + j];
            }
            C[i * numBColumns + j] = sum;
        }
    }
}

// Host function for matrix multiplication
void matrixMulHost(const float* h_A, const float* h_B, float* h_C, int numARows, int numAColumns, int numBColumns) {
    int sizeA = numARows * numAColumns * sizeof(float);
    int sizeB = numAColumns * numBColumns * sizeof(float);
    int sizeC = numARows * numBColumns * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((numBColumns - 1) / TILE_WIDTH + 1, (numARows - 1) / TILE_WIDTH + 1);
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, numAColumns, numBColumns);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Function to read matrix data from a file
bool readMatrixFromFile(const std::string& filename, float*& A, int& numARows, int& numAColumns,
                        float*& B, int& numBRows, int& numBColumns) {
    std::ifstream fileInput(filename);
    if (!fileInput.is_open()) {
        std::cerr << "Error: could not open file!" << std::endl;
        return false;
    }

    fileInput >> numARows >> numAColumns;
    A = new float[numARows * numAColumns];
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numAColumns; j++) {
            fileInput >> A[i * numAColumns + j];
        }
    }

    fileInput >> numBRows >> numBColumns;
    if (numAColumns != numBRows) {
        std::cerr << "Error: Incompatible matrix dimensions for multiplication!" << std::endl;
        delete[] A;
        return false;
    }

    B = new float[numBRows * numBColumns];
    for (int i = 0; i < numBRows; i++) {
        for (int j = 0; j < numBColumns; j++) {
            fileInput >> B[i * numBColumns + j];
        }
    }

    fileInput.close();
    return true;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return -1;
    }

    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    float* h_C_seq = nullptr;
    int numARows, numAColumns, numBRows, numBColumns;

    if (!readMatrixFromFile(argv[1], h_A, numARows, numAColumns, h_B, numBRows, numBColumns)) {
        return -1;
    }

    h_C = new float[numARows * numBColumns];
    h_C_seq = new float[numARows * numBColumns];

    clock_t start_time = clock();
    matrixMulHost(h_A, h_B, h_C, numARows, numAColumns, numBColumns);
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Tiempo para realizar multiplicación en CUDA: %.6f segundos\n", elapsed_time);

    start_time = clock();
    matrixMulSequential(h_A, h_B, h_C_seq, numARows, numAColumns, numBColumns);
    end_time = clock();
    elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Tiempo para realizar multiplicación secuencial: %.6f segundos\n", elapsed_time);

    bool match = true;
    for (int i = 0; i < numARows * numBColumns; i++) {
        if (fabs(h_C[i] - h_C_seq[i]) > 1e-4) {
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "Resultados coinciden entre CUDA y el cálculo secuencial." << std::endl;
    } else {
        std::cout << "Discrepancia entre resultados de CUDA y el cálculo secuencial." << std::endl;
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_seq;

    return 0;
}
//nvcc -o matrix_mul matrix_multiplication_cuda.cu