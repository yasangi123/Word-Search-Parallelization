// cuda_word_search.cu
// CUDA‐accelerated multi‐phrase search with interactive input
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <cuda_runtime.h>

#define MAX_LINE_LENGTH 8192    // max characters per line
#define MAX_PHRASES      100    // max number of search phrases
#define MAX_INPUT_SIZE   4096   // buffer for user input

// ——— host: normalize in-place ———————————————————————————————
// Removes punctuation, lowercases letters, keeps spaces.
static void normalize_host(char* s) {
    char* dst = s;
    for (char* src = s; *src; ++src) {
        unsigned char c = (unsigned char)*src;
        if (isalpha(c) || isspace(c))
            *dst++ = tolower(c);
    }
    *dst = '\0';
}

// ——— device: compute string length on the GPU ————————————————
__device__ int dev_strlen(const char *s) {
    int len = 0;
    while (s[len] != '\0') ++len;
    return len;
}

// ——— device: compare up to n chars on the GPU ————————————————
__device__ int dev_strncmp(const char *a, const char *b, int n) {
    for (int i = 0; i < n; ++i) {
        char ca = a[i], cb = b[i];
        if (ca != cb) return (int)ca - (int)cb;
        if (ca == '\0') return 0;
    }
    return 0;
}

// ——— device: naive phrase match with word-boundary checks ————————
__device__ bool phrase_match_device(const char* line, const char* phrase) {
    int L = dev_strlen(line), P = dev_strlen(phrase);
    if (P == 0 || P > L) return false;
    for (int i = 0; i + P <= L; ++i) {
        if (i > 0 && line[i-1] != ' ') continue;
        if (i+P < L && line[i+P] != ' ') continue;
        if (dev_strncmp(line + i, phrase, P) == 0) return true;
    }
    return false;
}

// ——— kernel: each thread processes one line, tests all phrases —————
__global__ void search_kernel(
    const char* __restrict__ d_lines,
    int line_count,
    const char* __restrict__ d_phrases,
    int phrase_count,
    int *d_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= line_count) return;
    const char* line_base   = d_lines   + (size_t)idx * MAX_LINE_LENGTH;
    const char* phrase_base = d_phrases                       ;
    for (int j = 0; j < phrase_count; ++j) {
        if (phrase_match_device(line_base, phrase_base + (size_t)j * MAX_LINE_LENGTH))
            atomicAdd(&d_counts[j], 1);
    }
}

// ——— perform CUDA search given user inputs —————————————————————
static int run_cuda_search(
    const char* filepath,
    char* phrases[],
    int phrase_count
) {
    // 1) Read & normalize all lines on host
    FILE* f = fopen(filepath, "r");
    if (!f) { perror("fopen"); return 1; }
    int cap = 1024, line_count = 0;
    char** lines = (char**)malloc(cap * sizeof(char*));
    char buf[MAX_LINE_LENGTH];
    while (fgets(buf, sizeof buf, f)) {
        if (line_count >= cap) {
            cap *= 2;
            lines = (char**)realloc(lines, cap * sizeof(char*));
            if (!lines) { fprintf(stderr, "OOM\n"); return 1; }
        }
        buf[strcspn(buf, "\r\n")] = '\0';
        normalize_host(buf);
        lines[line_count++] = strdup(buf);
    }
    fclose(f);

    // 2) Flatten lines into zeroed host buffer
    size_t lines_bytes = (size_t)line_count * MAX_LINE_LENGTH;
    char *h_lines_flat = (char*)calloc((size_t)line_count, MAX_LINE_LENGTH);
    for (int i = 0; i < line_count; ++i) {
        size_t L = strlen(lines[i]) + 1;
        memcpy(h_lines_flat + (size_t)i * MAX_LINE_LENGTH, lines[i], L);
    }

    // 3) Flatten phrases into zeroed host buffer
    size_t phrases_bytes = (size_t)phrase_count * MAX_LINE_LENGTH;
    char *h_phrases_flat = (char*)calloc((size_t)phrase_count, MAX_LINE_LENGTH);
    for (int j = 0; j < phrase_count; ++j) {
        size_t P = strlen(phrases[j]) + 1;
        memcpy(h_phrases_flat + (size_t)j * MAX_LINE_LENGTH, phrases[j], P);
    }

    // 4) Allocate device memory and copy
    char *d_lines, *d_phrases;
    int  *d_counts;
    cudaMalloc(&d_lines,   lines_bytes);
    cudaMalloc(&d_phrases, phrases_bytes);
    cudaMalloc(&d_counts,  phrase_count * sizeof(int));
    cudaMemset(d_counts, 0, phrase_count * sizeof(int));
    cudaMemcpy(d_lines,   h_lines_flat,   lines_bytes,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_phrases, h_phrases_flat, phrases_bytes, cudaMemcpyHostToDevice);

    // 5) Launch kernel
    int threads = 256;
    int blocks  = (line_count + threads - 1) / threads;
    cudaDeviceSynchronize();
    double t0 = clock() / (double)CLOCKS_PER_SEC;
    search_kernel<<<blocks,threads>>>(d_lines, line_count, d_phrases, phrase_count, d_counts);
    cudaDeviceSynchronize();
    double t1 = clock() / (double)CLOCKS_PER_SEC;

    // 6) Copy back counts and display
    int *h_counts = (int*)malloc(phrase_count * sizeof(int));
    cudaMemcpy(h_counts, d_counts, phrase_count * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n+-------------------------------+---------------+\n");
    printf("| %-29s | %13s |\n", "Phrase", "Matches");
    printf("+-------------------------------+---------------+\n");
    int total = 0;
    for (int j = 0; j < phrase_count; ++j) {
        printf("| %-29s | %13d |\n", phrases[j], h_counts[j]);
        total += h_counts[j];
    }
    printf("+-------------------------------+---------------+\n");
    printf("| %-29s | %13d |\n", "Total matches", total);
    printf("+-------------------------------+---------------+\n");
    printf("| %-29s | %13.4f |\n", "Elapsed time (s)", t1 - t0);
    printf("+-------------------------------+---------------+\n");

    // 7) Cleanup
    for (int i = 0; i < line_count; ++i) free(lines[i]);
    free(lines);
    free(h_lines_flat);
    free(h_phrases_flat);
    free(h_counts);
    cudaFree(d_lines);
    cudaFree(d_phrases);
    cudaFree(d_counts);
    return 0;
}

// ——— Main: interactive menu to get inputs ——————————————————————
int main(void) {
    char filepath[MAX_INPUT_SIZE];
    char phrase_line[MAX_INPUT_SIZE];
    char* phrases[MAX_PHRASES];
    int phrase_count = 0;

    // 1) Get file path
    printf("Enter path to text file: ");
    if (!fgets(filepath, sizeof filepath, stdin)) return 0;
    filepath[strcspn(filepath, "\r\n")] = '\0';

    // 2) Get comma-separated phrases
    printf("Enter search phrases, comma-separated:\n");
    if (!fgets(phrase_line, sizeof phrase_line, stdin)) return 0;
    phrase_line[strcspn(phrase_line, "\r\n")] = '\0';

    // 3) Split into phrases[]
    char* tok = strtok(phrase_line, ",");
    while (tok && phrase_count < MAX_PHRASES) {
        while (*tok == ' ') ++tok;  // trim leading
        char* end = tok + strlen(tok) - 1;
        while (end > tok && *end == ' ') *end-- = '\0';  // trim trailing
        if (*tok) phrases[phrase_count++] = tok;
        tok = strtok(NULL, ",");
    }
    if (phrase_count == 0) {
        printf("No valid phrases entered. Exiting.\n");
        return 0;
    }

    // 4) Run the CUDA‐accelerated search
    return run_cuda_search(filepath, phrases, phrase_count);
}
