// cuda.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <cuda_runtime.h>

#define MAX_LINE_LENGTH   8192   // max chars per line
#define MAX_PHRASES       100    // max number of phrases
#define MAX_INPUT_SIZE    4096   // interactive input buffer

// ——— host: normalize in-place ———————————————————————————————
// strip punctuation, lowercase letters, keep spaces
static void normalize_host(char* s) {
    char* dst = s;
    for (char* src = s; *src; ++src) {
        unsigned char c = (unsigned char)*src;
        if (isalpha(c) || isspace(c))
            *dst++ = tolower(c);
    }
    *dst = '\0';
}

// ——— device: strlen on GPU ————————————————————————————————
__device__ int dev_strlen(const char* s) {
    int n = 0;
    while (s[n]) ++n;
    return n;
}

// ——— device: strncmp on GPU ———————————————————————————————
__device__ int dev_strncmp(const char* a, const char* b, int n) {
    for (int i = 0; i < n; ++i) {
        char ca = a[i], cb = b[i];
        if (ca != cb) return (int)ca - (int)cb;
        if (!ca) return 0;
    }
    return 0;
}

// ——— device: sliding‐window whole‐word match ——————————————————
__device__ bool phrase_match_device(const char* line, const char* phrase) {
    int L = dev_strlen(line);
    int P = dev_strlen(phrase);
    if (P == 0 || P > L) return false;
    for (int i = 0; i + P <= L; ++i) {
        // word boundary before
        if (i > 0 && line[i-1] != ' ') continue;
        // word boundary after
        if (i+P < L && line[i+P] != ' ') continue;
        if (dev_strncmp(line + i, phrase, P) == 0)
            return true;
    }
    return false;
}

// ——— kernel: each thread processes one line —————————————————
__global__ void search_kernel(
    const char* d_lines,
    int line_count,
    const char* d_phrases,
    int phrase_count,
    int* d_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= line_count) return;
    const char* line  = d_lines   + (size_t)idx * MAX_LINE_LENGTH;
    for (int j = 0; j < phrase_count; ++j) {
        const char* phrase = d_phrases + (size_t)j * MAX_LINE_LENGTH;
        if (phrase_match_device(line, phrase))
            atomicAdd(&d_counts[j], 1);
    }
}

// ——— host: run the full search + logging —————————————————————
static int run_cuda_search(const char* filepath, char* phrases[], int pc) {
    // 1) Read & normalize lines
    FILE* f = fopen(filepath, "r");
    if (!f) { perror("fopen"); return 1; }
    int cap = 1024, line_count = 0;
    char** lines = (char**)malloc(cap * sizeof(char*));
    char buf[MAX_LINE_LENGTH];
    while (fgets(buf, sizeof buf, f)) {
        if (line_count >= cap) {
            cap *= 2;
            lines = (char**)realloc(lines, cap * sizeof(char*));
            if (!lines) { fprintf(stderr,"Out of memory\n"); return 1; }
        }
        buf[strcspn(buf, "\r\n")] = '\0';
        normalize_host(buf);
        lines[line_count++] = strdup(buf);
    }
    fclose(f);

    // 2) Flatten into zeroed host buffers
    size_t LB = (size_t)line_count   * MAX_LINE_LENGTH;
    size_t PB = (size_t)pc           * MAX_LINE_LENGTH;
    char* h_lines   = (char*)calloc(line_count,   MAX_LINE_LENGTH);
    char* h_phrases = (char*)calloc(pc,           MAX_LINE_LENGTH);
    for (int i = 0; i < line_count; ++i) {
        size_t L = strlen(lines[i]) + 1;
        memcpy(h_lines   + (size_t)i*MAX_LINE_LENGTH,   lines[i], L);
    }
    for (int j = 0; j < pc; ++j) {
        size_t P = strlen(phrases[j]) + 1;
        memcpy(h_phrases + (size_t)j*MAX_LINE_LENGTH, phrases[j], P);
    }

    // 3) Allocate & upload to device
    char *d_lines, *d_phrases;
    int  *d_counts;
    cudaMalloc(&d_lines,   LB);
    cudaMalloc(&d_phrases, PB);
    cudaMalloc(&d_counts,  pc * sizeof(int));
    cudaMemset(d_counts, 0, pc * sizeof(int));
    cudaMemcpy(d_lines,   h_lines,   LB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_phrases, h_phrases, PB, cudaMemcpyHostToDevice);

    // 4) Launch kernel
    int threads = 256;
    int blocks  = (line_count + threads - 1) / threads;
    cudaDeviceSynchronize();
    double t0 = clock()/(double)CLOCKS_PER_SEC;
    search_kernel<<<blocks,threads>>>(d_lines, line_count, d_phrases, pc, d_counts);
    cudaDeviceSynchronize();
    double t1 = clock()/(double)CLOCKS_PER_SEC;

    // 5) Download counts & print table
    int* h_counts = (int*)malloc(pc * sizeof(int));
    cudaMemcpy(h_counts, d_counts, pc * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n+-------------------------------+---------------+\n");
    printf("| %-29s | %13s |\n", "Phrase", "Matches");
    printf("+-------------------------------+---------------+\n");
    int total = 0;
    for (int j = 0; j < pc; ++j) {
        printf("| %-29s | %13d |\n", phrases[j], h_counts[j]);
        total += h_counts[j];
    }
    printf("+-------------------------------+---------------+\n");
    printf("| %-29s | %13d |\n", "Total matches", total);
    printf("+-------------------------------+---------------+\n");
    printf("| %-29s | %13.4f |\n", "Elapsed time (s)", t1 - t0);
    printf("+-------------------------------+---------------+\n");

    // 6) Append to CSV
    {
        static int hdr = 0;
        FILE* csv = fopen("results.csv","a");
        if (csv) {
            if (!hdr) {
                fprintf(csv,"timestamp,filename,phrases,total_matches,time_s\n");
                hdr = 1;
            }
            char plist[MAX_INPUT_SIZE] = "";
            for (int j = 0; j < pc; ++j) {
                if (j) strcat(plist, ";");
                strcat(plist, phrases[j]);
            }
            char tbuf[64];
            time_t now = time(NULL);
            strftime(tbuf,sizeof tbuf,"%Y-%m-%d %H:%M:%S", localtime(&now));
            fprintf(csv,"\"%s\",\"%s\",\"%s\",%d,%.4f\n",
                    tbuf, filepath, plist, total, t1 - t0);
            fclose(csv);
        }
    }

    // 7) Clean up
    for (int i = 0; i < line_count; ++i) free(lines[i]);
    free(lines);
    free(h_lines);
    free(h_phrases);
    free(h_counts);
    cudaFree(d_lines);
    cudaFree(d_phrases);
    cudaFree(d_counts);
    return 0;
}

// ——— main: interactive prompts ——————————————————————————————
int main(void) {
    char filepath[MAX_INPUT_SIZE];
    char line[MAX_INPUT_SIZE];
    char* phrases[MAX_PHRASES];
    int pc = 0;

    printf("Enter path to text file: ");
    if (!fgets(filepath,sizeof filepath,stdin)) return 0;
    filepath[strcspn(filepath,"\r\n")] = '\0';

    printf("Enter search phrases, comma-separated:\n");
    if (!fgets(line,sizeof line,stdin)) return 0;
    line[strcspn(line,"\r\n")] = '\0';

    char* tok = strtok(line, ",");
    while (tok && pc < MAX_PHRASES) {
        while (*tok==' ') ++tok;
        char* end = tok + strlen(tok) - 1;
        while (end>tok && *end==' ') *end-- = '\0';
        if (*tok) phrases[pc++] = tok;
        tok = strtok(NULL, ",");
    }
    if (pc == 0) {
        printf("No valid phrases entered. Exiting.\n");
        return 0;
    }

    return run_cuda_search(filepath, phrases, pc);
}
