// cuda.cu
// CUDA-accelerated phrase search: read a text file, normalize/tokenize on host,
// then in parallel scan each line for all phrases on the GPU.
// Usage: nvcc -O2 cuda.cu -o cuda && ./cuda [threadsPerBlock]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

#define MAX_INPUT_SIZE 4096

// normalize in-place: keep only letters/spaces, lowercase
void normalize(std::string &s) {
    size_t dst = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = (unsigned char)s[i];
        if (isalpha(c) || isspace(c))
            s[dst++] = tolower(c);
    }
    s.resize(dst);
}

// split comma-separated phrases (host)
std::vector<std::string> split_phrases(const std::string &line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t a = tok.find_first_not_of(" \t");
        size_t b = tok.find_last_not_of(" \t");
        if (a != std::string::npos)
            out.push_back(tok.substr(a, b - a + 1));
    }
    return out;
}

// GPU kernel: each thread scans one line against all phrases
__global__ void searchKernel(
    const char* d_lines,
    const size_t* d_lineOff,
    const int*    d_lineLen,
    const char*   d_phrases,
    const size_t* d_phrOff,
    const int*    d_phrLen,
    int lineCount,
    int phrCount,
    int* d_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lineCount) return;

    const char* line = d_lines + d_lineOff[idx];
    int L = d_lineLen[idx];

    for (int p = 0; p < phrCount; ++p) {
        const char* phr = d_phrases + d_phrOff[p];
        int P = d_phrLen[p];
        for (int i = 0; i + P <= L; ++i) {
            // ensure word boundaries
            if ((i > 0 && line[i-1] != ' ') || (i+P < L && line[i+P] != ' '))
                continue;
            bool ok = true;
            for (int j = 0; j < P; ++j) {
                if (line[i+j] != phr[j]) { ok = false; break; }
            }
            if (ok) {
                atomicAdd(&d_counts[p], 1);
                break;
            }
        }
    }
}

int main(int argc, char** argv) {
    int threadsPerBlock = 256;
    if (argc > 1) {
        int t = atoi(argv[1]);
        if (t > 0) threadsPerBlock = t;
    }

    // — host input —
    char filepath[MAX_INPUT_SIZE];
    printf("Enter path to text file: ");
    if (!fgets(filepath, sizeof filepath, stdin)) return 1;
    filepath[strcspn(filepath, "\r\n")] = '\0';

    char phrase_line[MAX_INPUT_SIZE];
    printf("Enter search phrases, comma-separated:\n");
    if (!fgets(phrase_line, sizeof phrase_line, stdin)) return 1;
    phrase_line[strcspn(phrase_line, "\r\n")] = '\0';

    auto phrases = split_phrases(phrase_line);
    int pc = (int)phrases.size();
    if (pc == 0) {
        fprintf(stderr, "No valid phrases entered. Exiting.\n");
        return 1;
    }

    // — read & normalize file lines —
    std::ifstream infile(filepath);
    if (!infile) {
        perror("Error opening file");
        return 1;
    }
    std::vector<std::string> lines;
    std::string raw;
    while (std::getline(infile, raw)) {
        normalize(raw);
        if (!raw.empty()) lines.push_back(raw);
    }
    infile.close();
    int LC = (int)lines.size();
    if (LC == 0) {
        fprintf(stderr, "No lines to process. Exiting.\n");
        return 1;
    }

    // — build flat buffers & offsets —
    std::vector<size_t> lineOff(LC), phrOff(pc);
    std::vector<int>    lineLen(LC),  phrLen(pc);
    size_t totL = 0, totP = 0;
    for (int i = 0; i < LC; ++i) {
        lineOff[i] = totL;
        lineLen[i] = (int)lines[i].size();
        totL += lines[i].size();
    }
    for (int p = 0; p < pc; ++p) {
        normalize(phrases[p]);
        phrOff[p] = totP;
        phrLen[p] = (int)phrases[p].size();
        totP += phrases[p].size();
    }

    std::vector<char> bufL(totL), bufP(totP);
    for (int i = 0; i < LC; ++i)
        memcpy(&bufL[lineOff[i]], lines[i].data(), lineLen[i]);
    for (int p = 0; p < pc; ++p)
        memcpy(&bufP[phrOff[p]], phrases[p].data(), phrLen[p]);

    // — device alloc & copy —
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
    cudaMemset(dC, 0, pc * sizeof(int));

    cudaMemcpy(dL,  bufL.data(),         totL,             cudaMemcpyHostToDevice);
    cudaMemcpy(dLO, lineOff.data(),      LC * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dLL, lineLen.data(),      LC * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(dP,  bufP.data(),         totP,             cudaMemcpyHostToDevice);
    cudaMemcpy(dPO, phrOff.data(),       pc * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dPL, phrLen.data(),       pc * sizeof(int),    cudaMemcpyHostToDevice);

    // — kernel launch —
    int blocks = (LC + threadsPerBlock - 1) / threadsPerBlock;
    clock_t t0 = clock();
    searchKernel<<<blocks, threadsPerBlock>>>(
        dL, dLO, dLL,
        dP, dPO, dPL,
        LC, pc, dC
    );
    cudaDeviceSynchronize();
    clock_t t1 = clock();

    // — copy back counts —
    std::vector<int> counts(pc);
    cudaMemcpy(counts.data(), dC, pc * sizeof(int), cudaMemcpyDeviceToHost);

    double elapsed = double(t1 - t0) / CLOCKS_PER_SEC;
    int totalMatches = 0;

    // — print results —
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
    printf("| %-29s | %13.4f |\n", "Elapsed time (s)", elapsed);
    printf("+-------------------------------+---------------+\n");

    // — append to CSV —
    {
        static bool header_done = false;
        std::ofstream csv("results.csv", std::ios::app);
        if (csv) {
            if (!header_done) {
                csv << "timestamp,filename,phrases,total_matches,time_s\n";
                header_done = true;
            }
            time_t now = time(NULL);
            char tbuf[64];
            strftime(tbuf, sizeof tbuf, "%Y-%m-%d %H:%M:%S", localtime(&now));
            csv << '"' << tbuf << "\",\""
                << filepath << "\",\"";
            for (int i = 0; i < pc; ++i) {
                if (i) csv << ';';
                csv << phrases[i];
            }
            csv << "\"," << totalMatches << ',' << elapsed << "\n";
        }
    }

    // — cleanup —
    cudaFree(dL);
    cudaFree(dLO);
    cudaFree(dLL);
    cudaFree(dP);
    cudaFree(dPO);
    cudaFree(dPL);
    cudaFree(dC);

    return 0;
}
