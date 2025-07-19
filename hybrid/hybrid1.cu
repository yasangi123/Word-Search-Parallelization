#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <omp.h>

#define MAX_LINES           200000
#define MAX_PHRASES         100
#define MAX_LINE_LENGTH     8192
#define MAX_INPUT_SIZE      4096

// GPU kernel: one thread per line, naive phrase search with word-boundary checks
__global__ void gpuSearchKernel(
    const char *d_lines,
    const size_t *d_offsets,
    const int *d_lineLens,
    const char *d_phrases,
    const size_t *d_pOffsets,
    const int *d_pLens,
    int lineCount,
    int phraseCount,
    int *d_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lineCount) return;

    const char *line = d_lines + d_offsets[idx];
    int llen = d_lineLens[idx];

    // for each phrase, scan this line
    for (int p = 0; p < phraseCount; ++p) {
        const char *phrase = d_phrases + d_pOffsets[p];
        int plen = d_pLens[p];
        bool found = false;

        // naive substring search with boundary checks
        for (int i = 0; i + plen <= llen; ++i) {
            if (i > 0 && line[i-1] != ' ') continue;
            if (i+plen < llen && line[i+plen] != ' ') continue;
            bool ok = true;
            for (int j = 0; j < plen; ++j) {
                if (line[i+j] != phrase[j]) { ok = false; break; }
            }
            if (ok) { found = true; break; }
        }

        if (found) {
            // atomic increment to avoid race on same d_counts[p]
            atomicAdd(&d_counts[p], 1);
        }
    }
}

// Normalize string in-place: keep only letters & spaces, convert to lowercase
void normalize(char *s) {
    char *dst = s;
    for (char *src = s; *src; ++src) {
        unsigned char c = (unsigned char)*src;
        if (isalpha(c) || isspace(c)) {
            *dst++ = tolower(c);
        }
    }
    *dst = '\0';
}

int main(int argc, char **argv) {
    char filepath[MAX_INPUT_SIZE];
    char phrase_line[MAX_INPUT_SIZE];

    // read file path and phrases from stdin
    printf("Enter path to text file: ");
    if (!fgets(filepath, sizeof(filepath), stdin)) return 0;
    filepath[strcspn(filepath, "\r\n")] = '\0';

    printf("Enter search phrases, comma-separated:\n");
    if (!fgets(phrase_line, sizeof(phrase_line), stdin)) return 0;
    phrase_line[strcspn(phrase_line, "\r\n")] = '\0';

    // --- CPU: load & normalize lines ---
    std::vector<std::string> lines;
    lines.reserve(MAX_LINES);
    FILE *f = fopen(filepath, "r");
    if (!f) { perror("Error opening file"); return 1; }
    char buf[MAX_LINE_LENGTH];
    while (lines.size() < MAX_LINES && fgets(buf, sizeof(buf), f)) {
        normalize(buf);
        lines.emplace_back(buf);
    }
    fclose(f);
    int lineCount = (int)lines.size();
    if (!lineCount) { printf("No lines read.\n"); return 0; }

    // --- CPU: parse & normalize phrases ---
    std::vector<std::string> phrases;
    phrases.reserve(MAX_PHRASES);
    char *tok = strtok(phrase_line, ",");
    while (tok && phrases.size() < MAX_PHRASES) {
        while (*tok == ' ') ++tok;
        char *end = tok + strlen(tok) - 1;
        while (end > tok && *end == ' ') *end-- = '\0';
        if (*tok) {
            normalize(tok);
            phrases.emplace_back(tok);
        }
        tok = strtok(nullptr, ",");
    }
    int phraseCount = (int)phrases.size();
    if (!phraseCount) return 0;

    // --- CPU: flatten lines into one buffer + offsets ---
    std::vector<size_t> offsets(lineCount);
    size_t totalChars = 0;
    for (int i = 0; i < lineCount; ++i) {
        offsets[i] = totalChars;
        totalChars += lines[i].size();
    }
    std::vector<char> h_lines(totalChars);
    std::vector<int>  h_lineLens(lineCount);
    totalChars = 0;
    for (int i = 0; i < lineCount; ++i) {
        memcpy(&h_lines[totalChars], lines[i].data(), lines[i].size());
        h_lineLens[i] = (int)lines[i].size();
        totalChars += lines[i].size();
    }

    // --- CPU: flatten phrases into one buffer + offsets ---
    std::vector<size_t> pOffsets(phraseCount);
    size_t totalP = 0;
    for (int p = 0; p < phraseCount; ++p) {
        pOffsets[p] = totalP;
        totalP += phrases[p].size();
    }
    std::vector<char> h_phrases(totalP);
    std::vector<int>  h_pLens(phraseCount);
    totalP = 0;
    for (int p = 0; p < phraseCount; ++p) {
        memcpy(&h_phrases[totalP], phrases[p].data(), phrases[p].size());
        h_pLens[p] = (int)phrases[p].size();
        totalP += phrases[p].size();
    }

    // --- CPU → GPU: allocate and copy buffers ---
    char   *d_lines;
    size_t *d_offsets;
    int    *d_lineLens;
    char   *d_phrases;
    size_t *d_pOffsets;
    int    *d_pLens;
    int    *d_counts;

    cudaMalloc(&d_lines,    totalChars);
    cudaMalloc(&d_offsets,  lineCount  * sizeof(size_t));
    cudaMalloc(&d_lineLens, lineCount  * sizeof(int));
    cudaMalloc(&d_phrases,  totalP);
    cudaMalloc(&d_pOffsets, phraseCount * sizeof(size_t));
    cudaMalloc(&d_pLens,    phraseCount * sizeof(int));
    cudaMalloc(&d_counts,   phraseCount * sizeof(int));

    cudaMemcpy(d_lines,    h_lines.data(),   totalChars,                         cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets,  offsets.data(),   lineCount  * sizeof(size_t),        cudaMemcpyHostToDevice);
    cudaMemcpy(d_lineLens, h_lineLens.data(),lineCount  * sizeof(int),           cudaMemcpyHostToDevice);
    cudaMemcpy(d_phrases,  h_phrases.data(), totalP,                             cudaMemcpyHostToDevice);
    cudaMemcpy(d_pOffsets, pOffsets.data(),  phraseCount * sizeof(size_t),       cudaMemcpyHostToDevice);
    cudaMemcpy(d_pLens,    h_pLens.data(),   phraseCount * sizeof(int),          cudaMemcpyHostToDevice);
    cudaMemset(d_counts,   0,                phraseCount * sizeof(int));  // zero counts

    // configure GPU & CPU parallelism
    int tp   = (argc>1 ? atoi(argv[1]) : 256);
    int ompT = (argc>2 ? atoi(argv[2]) : omp_get_max_threads());
    omp_set_num_threads(ompT);
    printf("Using %d CUDA threads per block\n", tp);
    printf("Using %d OpenMP threads\n", ompT);

    // --- GPU execution ---
    int blocks = (lineCount + tp - 1) / tp;
    clock_t t0 = clock();
    gpuSearchKernel<<<blocks, tp>>>(
        d_lines, d_offsets, d_lineLens,
        d_phrases, d_pOffsets, d_pLens,
        lineCount, phraseCount, d_counts
    );
    cudaDeviceSynchronize();
    clock_t t1 = clock();
    double secs = double(t1 - t0) / CLOCKS_PER_SEC;

    // --- GPU → CPU: retrieve results ---
    std::vector<int> h_counts(phraseCount);
    cudaMemcpy(h_counts.data(), d_counts,
               phraseCount * sizeof(int),
               cudaMemcpyDeviceToHost);

    // --- CPU: print results ---
    printf("\n+-------------------------------+---------------+\n");
    printf("| %-29s | %13s |\n","Phrase","Matches");
    printf("+-------------------------------+---------------+\n");
    int total = 0;
    for (int p = 0; p < phraseCount; ++p) {
        printf("| %-29s | %13d |\n", phrases[p].c_str(), h_counts[p]);
        total += h_counts[p];
    }
    printf("+-------------------------------+---------------+\n");
    printf("| %-29s | %13d |\n","Total matches", total);
    printf("+-------------------------------+---------------+\n");
    printf("| %-29s | %13.4f |\n","Elapsed time (s)", secs);
    printf("+-------------------------------+---------------+\n");

    // --- Cleanup GPU memory ---
    cudaFree(d_lines);
    cudaFree(d_offsets);
    cudaFree(d_lineLens);
    cudaFree(d_phrases);
    cudaFree(d_pOffsets);
    cudaFree(d_pLens);
    cudaFree(d_counts);

    return 0;
}
