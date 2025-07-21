// hybrid.cu

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <cuda_runtime.h>

#define MAX_INPUT_SIZE 4096  // Max length for file path and input phrases

// normalize in-place: keep only letters/spaces, convert to lowercase
void normalize(std::string &s) {
    size_t dst = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = (unsigned char)s[i];
        if (isalpha(c) || isspace(c)) {
            s[dst++] = tolower(c);
        }
        // Other chars dropped
    }
    s.resize(dst);  // Truncate to new length
}

// Split comma-separated phrases, trim whitespace
std::vector<std::string> split_phrases(const std::string &line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t a = tok.find_first_not_of(" \t");
        size_t b = tok.find_last_not_of(" \t");
        if (a != std::string::npos) {
            out.push_back(tok.substr(a, b - a + 1));
        }
    }
    return out;
}

// GPU kernel: each thread processes one line, compares against all phrases
__global__ void searchKernel(
    const char*   d_lines,    // Flattened buffer of all lines
    const size_t* d_lineOff,  // Offsets into d_lines for each line
    const int*    d_lineLen,  // Lengths of each line
    const char*   d_phrases,  // Flattened buffer of all phrases
    const size_t* d_phrOff,   // Offsets into d_phrases for each phrase
    const int*    d_phrLen,   // Lengths of each phrase
    int           lineCount,  // Total number of lines
    int           phrCount,   // Total number of phrases
    int*          d_counts    // Output count of matches per phrase
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lineCount) return;  // Out-of-range threads exit

    // Pointer and length for this thread's line
    const char* line = d_lines + d_lineOff[idx];
    int L = d_lineLen[idx];

    // Loop over each phrase
    for (int p = 0; p < phrCount; ++p) {
        const char* phr = d_phrases + d_phrOff[p];
        int P = d_phrLen[p];

        // Slide a window of length P over the line
        for (int i = 0; i + P <= L; ++i) {
            // Word-boundary check: must be space or edge
            if ((i > 0 && line[i-1] != ' ') || (i + P < L && line[i+P] != ' '))
                continue;

            // Character-by-character compare
            bool ok = true;
            for (int j = 0; j < P; ++j) {
                if (line[i+j] != phr[j]) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                atomicAdd(&d_counts[p], 1);  // Count one match
                break;  // Stop scanning this phrase in this line
            }
        }
    }
}

int main(int argc, char** argv) {
    // Determine threads per block (default 256)
    int threadsPerBlock = 256;
    if (argc > 1) {
        int t = atoi(argv[1]);
        if (t > 0) threadsPerBlock = t;
    }

    // — Host input: file path —
    char filepath[MAX_INPUT_SIZE];
    printf("Enter path to text file: ");
    if (!fgets(filepath, sizeof filepath, stdin)) return 1;
    filepath[strcspn(filepath, "\r\n")] = '\0';  // Strip newline

    // — Host input: phrases —
    char phrase_line[MAX_INPUT_SIZE];
    printf("Enter search phrases, comma-separated:\n");
    if (!fgets(phrase_line, sizeof phrase_line, stdin)) return 1;
    phrase_line[strcspn(phrase_line, "\r\n")] = '\0';

    // Split and store phrases
    auto phrases = split_phrases(phrase_line);
    int pc = (int)phrases.size();
    if (pc == 0) {
        fprintf(stderr, "No valid phrases entered. Exiting.\n");
        return 1;
    }

    // — Read lines from file —
    std::vector<std::string> lines;
    {
        std::ifstream infile(filepath);
        if (!infile) { perror("Error opening file"); return 1; }
        std::string raw;
        while (std::getline(infile, raw))
            lines.push_back(raw);
    }
    int LC = (int)lines.size();
    if (LC == 0) { fprintf(stderr, "No lines to process. Exiting.\n"); return 1; }

    // — Parallel normalize lines using OpenMP —
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < LC; ++i) {
        normalize(lines[i]);
    }

    // — Parallel normalize phrases —
    #pragma omp parallel for
    for (int p = 0; p < pc; ++p) {
        normalize(phrases[p]);
    }

    // Prepare offset/length vectors
    std::vector<size_t> lineOff(LC), phrOff(pc);
    std::vector<int>    lineLen(LC), phrLen(pc);
    size_t totL = 0, totP = 0;

    // — Compute total lengths in parallel (reduction) —
    #pragma omp parallel for reduction(+:totL)
    for (int i = 0; i < LC; ++i) {
        lineLen[i] = (int)lines[i].size();
        totL += lineLen[i];
    }
    #pragma omp parallel for reduction(+:totP)
    for (int p = 0; p < pc; ++p) {
        phrLen[p] = (int)phrases[p].size();
        totP += phrLen[p];
    }

    // — Compute offsets with serial prefix-sum —
    for (int i = 0; i < LC; ++i)
        lineOff[i] = (i == 0 ? 0 : lineOff[i-1] + lineLen[i-1]);
    for (int p = 0; p < pc; ++p)
        phrOff[p]  = (p == 0 ? 0 : phrOff[p-1]  + phrLen[p-1]);

    // Allocate flat character buffers
    std::vector<char> bufL(totL), bufP(totP);

    // — Copy lines and phrases into flat buffers in parallel —
    #pragma omp parallel for
    for (int i = 0; i < LC; ++i)
        memcpy(&bufL[lineOff[i]], lines[i].data(), lineLen[i]);
    #pragma omp parallel for
    for (int p = 0; p < pc; ++p)
        memcpy(&bufP[phrOff[p]], phrases[p].data(), phrLen[p]);

    // — Allocate device memory —
    char   *dL, *dP;
    size_t *dLO, *dPO;
    int    *dLL, *dPL, *dC;
    cudaMalloc(&dL,  totL);
    cudaMalloc(&dLO, LC * sizeof(size_t));
    cudaMalloc(&dLL, LC * sizeof(int));
    cudaMalloc(&dP,  totP);
    cudaMalloc(&dPO, pc * sizeof(size_t));
    cudaMalloc(&dPL, pc * sizeof(int));
    cudaMalloc(&dC,  pc * sizeof(int));
    cudaMemset( dC, 0,   pc * sizeof(int));  // Initialize counts

    // — Copy data to device —
    cudaMemcpy(dL,  bufL.data(),       totL,             cudaMemcpyHostToDevice);
    cudaMemcpy(dLO, lineOff.data(),    LC * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dLL, lineLen.data(),    LC * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(dP,  bufP.data(),       totP,             cudaMemcpyHostToDevice);
    cudaMemcpy(dPO, phrOff.data(),     pc * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dPL, phrLen.data(),     pc * sizeof(int),    cudaMemcpyHostToDevice);

    // — Launch GPU kernel and time it —
    int blocks = (LC + threadsPerBlock - 1) / threadsPerBlock;
    double t0 = omp_get_wtime();
    searchKernel<<<blocks, threadsPerBlock>>>(
        dL, dLO, dLL,
        dP, dPO, dPL,
        LC, pc, dC
    );
    cudaDeviceSynchronize();
    double t1 = omp_get_wtime();

    // — Retrieve counts from device —
    std::vector<int> counts(pc);
    cudaMemcpy(counts.data(), dC, pc * sizeof(int), cudaMemcpyDeviceToHost);

    // — Output results —
    double elapsed = t1 - t0;
    int totalMatches = 0;
    printf("\n+-------------------------------+---------------+\n");
    printf("| %-29s | %13s |\n", "Phrase", "Matches");
    printf("+-------------------------------+---------------+\n");
    for (int i = 0; i < pc; ++i) {
        printf("| %-29s | %13d |\n", phrases[i].c_str(), counts[i]);
        totalMatches += counts[i];
    }
    printf("+-------------------------------+---------------+\n");
    printf("| %-29s | %13d |\n", "Total matches", totalMatches);
    printf("+-------------------------------+---------------+\n");
    printf("| %-29s | %13.6f |\n", "Elapsed time (s)", elapsed);
    printf("+-------------------------------+---------------+\n");

    // — Cleanup device memory —
    cudaFree(dL);
    cudaFree(dLO);
    cudaFree(dLL);
    cudaFree(dP);
    cudaFree(dPO);
    cudaFree(dPL);
    cudaFree(dC);

    return 0;
}
