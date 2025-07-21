// cuda.cu

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

#define MAX_INPUT_SIZE 4096  // maximum length for file path and phrase line

// normalize in-place: keep only letters and spaces, convert to lowercase
void normalize(std::string &s) {
    size_t dst = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = (unsigned char)s[i];
        if (isalpha(c) || isspace(c)) {
            // copy valid char into output position
            s[dst++] = tolower(c);
        }
        // all other characters are dropped
    }
    s.resize(dst);  // truncate to new length
}

// split a comma-separated line into trimmed phrases
std::vector<std::string> split_phrases(const std::string &line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string tok;
    // extract tokens separated by ','
    while (std::getline(ss, tok, ',')) {
        // trim leading/trailing whitespace
        size_t a = tok.find_first_not_of(" \t");
        size_t b = tok.find_last_not_of(" \t");
        if (a != std::string::npos) {
            out.push_back(tok.substr(a, b - a + 1));
        }
    }
    return out;
}

// GPU kernel: each thread scans one line string against all phrases
__global__ void searchKernel(
    const char*   d_lines,    // flattened all-lines char buffer
    const size_t* d_lineOff,  // per-line offset into d_lines
    const int*    d_lineLen,  // per-line length
    const char*   d_phrases,  // flattened all-phrases char buffer
    const size_t* d_phrOff,   // per-phrase offset into d_phrases
    const int*    d_phrLen,   // per-phrase length
    int           lineCount,  // total number of lines
    int           phrCount,   // total number of phrases
    int*          d_counts    // output counters per phrase
) {
    // compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lineCount) return;  // outside range → done

    // pointer & length for this thread’s line
    const char* line = d_lines + d_lineOff[idx];
    int L = d_lineLen[idx];

    // for each phrase p
    for (int p = 0; p < phrCount; ++p) {
        const char* phr = d_phrases + d_phrOff[p];
        int P = d_phrLen[p];

        // slide a window of length P over the line
        for (int i = 0; i + P <= L; ++i) {
            // enforce whole-word matching: must be bounded by spaces or edges
            if ((i > 0 && line[i-1] != ' ') ||
                (i+P < L && line[i+P] != ' '))
            {
                continue;
            }

            // compare P characters
            bool ok = true;
            for (int j = 0; j < P; ++j) {
                if (line[i+j] != phr[j]) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                // atomically count one match for this phrase
                atomicAdd(&d_counts[p], 1);
                break;  // stop scanning this line for phrase p
            }
        }
    }
}

int main(int argc, char** argv) {
    // parse optional threads-per-block from argv
    int threadsPerBlock = 256;
    if (argc > 1) {
        int t = atoi(argv[1]);
        if (t > 0) threadsPerBlock = t;
    }

    // — host input: file path —
    char filepath[MAX_INPUT_SIZE];
    printf("Enter path to text file: ");
    if (!fgets(filepath, sizeof filepath, stdin)) return 1;
    filepath[strcspn(filepath, "\r\n")] = '\0';  // strip newline

    // — host input: comma-separated phrases —
    char phrase_line[MAX_INPUT_SIZE];
    printf("Enter search phrases, comma-separated:\n");
    if (!fgets(phrase_line, sizeof phrase_line, stdin)) return 1;
    phrase_line[strcspn(phrase_line, "\r\n")] = '\0';

    // split & trim phrases
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
        normalize(raw);            // keep letters/spaces, lowercase
        if (!raw.empty())          // skip empty after normalize
            lines.push_back(raw);
    }
    infile.close();
    int LC = (int)lines.size();
    if (LC == 0) {
        fprintf(stderr, "No lines to process. Exiting.\n");
        return 1;
    }

    // — build flattened char buffers & offset/length arrays —
    std::vector<size_t> lineOff(LC), phrOff(pc);
    std::vector<int>    lineLen(LC), phrLen(pc);
    size_t totL = 0, totP = 0;

    // compute offsets & total char counts for lines
    for (int i = 0; i < LC; ++i) {
        lineOff[i] = totL;
        lineLen[i] = (int)lines[i].size();
        totL += lineLen[i];
    }
    // compute offsets & total char counts for phrases
    for (int p = 0; p < pc; ++p) {
        normalize(phrases[p]);
        phrOff[p] = totP;
        phrLen[p] = (int)phrases[p].size();
        totP += phrLen[p];
    }

    // allocate and fill flat char buffers
    std::vector<char> bufL(totL), bufP(totP);
    for (int i = 0; i < LC; ++i)
        memcpy(&bufL[lineOff[i]], lines[i].data(), lineLen[i]);
    for (int p = 0; p < pc; ++p)
        memcpy(&bufP[phrOff[p]], phrases[p].data(), phrLen[p]);

    // — device allocation & copy host→device —
    char   *dL, *dP;
    size_t *dLO, *dPO;
    int    *dLL, *dPL, *dC;

    cudaMalloc(&dL,  totL);                // all lines
    cudaMalloc(&dLO, LC * sizeof(size_t)); // line offsets
    cudaMalloc(&dLL, LC * sizeof(int));    // line lengths
    cudaMalloc(&dP,  totP);                // all phrases
    cudaMalloc(&dPO, pc * sizeof(size_t)); // phrase offsets
    cudaMalloc(&dPL, pc * sizeof(int));    // phrase lengths
    cudaMalloc(&dC,  pc * sizeof(int));    // match counts
    cudaMemset(dC, 0, pc * sizeof(int));   // initialize counters to zero

    // copy buffers & metadata
    cudaMemcpy(dL,  bufL.data(),       totL,             cudaMemcpyHostToDevice);
    cudaMemcpy(dLO, lineOff.data(),    LC * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dLL, lineLen.data(),    LC * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(dP,  bufP.data(),       totP,             cudaMemcpyHostToDevice);
    cudaMemcpy(dPO, phrOff.data(),     pc * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dPL, phrLen.data(),     pc * sizeof(int),    cudaMemcpyHostToDevice);

    // — launch kernel and time it —
    int blocks = (LC + threadsPerBlock - 1) / threadsPerBlock;
    clock_t t0 = clock();
    searchKernel<<<blocks, threadsPerBlock>>>(
        dL, dLO, dLL,
        dP, dPO, dPL,
        LC, pc, dC
    );
    cudaDeviceSynchronize();
    clock_t t1 = clock();

    // — copy results back and print —
    std::vector<int> counts(pc);
    cudaMemcpy(counts.data(), dC, pc * sizeof(int), cudaMemcpyDeviceToHost);

    double elapsed = double(t1 - t0) / CLOCKS_PER_SEC;
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
    printf("| %-29s | %13.4f |\n", "Elapsed time (s)", elapsed);
    printf("+-------------------------------+---------------+\n");

    // — append summary to CSV (with one-time header) —
    {
        static bool header_done = false;
        std::ofstream csv("results.csv", std::ios::app);
        if (csv) {
            if (!header_done) {
                csv << "timestamp,filename,phrases,total_matches,time_s\n";
                header_done = true;
            }
            // format current time
            time_t now = time(NULL);
            char tbuf[64];
            strftime(tbuf, sizeof tbuf, "%Y-%m-%d %H:%M:%S", localtime(&now));
            // write CSV row
            csv << '"' << tbuf << "\",\""
                << filepath << "\",\"";
            for (int i = 0; i < pc; ++i) {
                if (i) csv << ';';
                csv << phrases[i];
            }
            csv << "\"," << totalMatches << ',' << elapsed << "\n";
        }
    }

    // — cleanup GPU memory —
    cudaFree(dL);
    cudaFree(dLO);
    cudaFree(dLL);
    cudaFree(dP);
    cudaFree(dPO);
    cudaFree(dPL);
    cudaFree(dC);

    return 0;
}
