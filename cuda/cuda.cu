
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <vector>
#include <string>
#include <unordered_map>
#include <cuda_runtime.h>

#define MAX_LINES             200000
#define MAX_PHRASES           100
#define MAX_TOKENS_PER_LINE   512
#define MAX_TOKENS_PER_PHRASE 64
#define MAX_INPUT_SIZE        4096
#define CSV_FILENAME          "results.csv"

// Device‐side constants
__constant__ int d_lineCount;
__constant__ int d_phraseCount;
__constant__ int d_maxLineTokens;
__constant__ int d_maxPhraseTokens;

// GPU kernel: one thread per line, integer‐ID compare
__global__ void tokenSearchKernel(
    const int *lineTokens,    // [lineCount][maxLineTokens]
    const int *lineLens,      // [lineCount]
    const int *phraseTokens,  // [phraseCount][maxPhraseTokens]
    const int *phraseLens,    // [phraseCount]
    int *counts               // [phraseCount]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_lineCount) return;

    const int *lt = lineTokens + idx * d_maxLineTokens;
    int llen = lineLens[idx];

    for (int p = 0; p < d_phraseCount; ++p) {
        const int *pt = phraseTokens + p * d_maxPhraseTokens;
        int plen = phraseLens[p];
        bool found = false;
        for (int i = 0; i + plen <= llen; ++i) {
            bool ok = true;
            #pragma unroll
            for (int j = 0; j < plen; ++j) {
                if (lt[i+j] != pt[j]) { ok = false; break; }
            }
            if (ok) { found = true; break; }
        }
        if (found) atomicAdd(&counts[p], 1);
    }
}

int main(int argc, char** argv) {
    // 1) Read inputs
    char filepath[MAX_INPUT_SIZE], phrase_line[MAX_INPUT_SIZE];
    printf("Enter path to text file: ");
    if (!fgets(filepath,sizeof filepath,stdin)) return 0;
    filepath[strcspn(filepath,"\r\n")] = '\0';

    printf("Enter search phrases, comma-separated:\n");
    if (!fgets(phrase_line,sizeof phrase_line,stdin)) return 0;
    phrase_line[strcspn(phrase_line,"\r\n")] = '\0';

    // 2) CPU normalize & tokenize words→IDs
    std::unordered_map<std::string,int> vocab;
    vocab.reserve(200000);
    int nextId = 1;  // reserve 0 for padding
    auto normalize = [&](std::string &s){
        for (auto &c:s) {
            if (isalpha((unsigned char)c)) c = tolower(c);
            else if (isspace((unsigned char)c)) /* keep */;
            else c = ' ';  // drop like CPU normalize
        }
    };
    auto tokenizeIds = [&](const std::string &s){
        std::vector<int> ids;
        ids.reserve(MAX_TOKENS_PER_LINE);
        size_t i=0,n=s.size();
        while (i<n) {
            while (i<n && isspace((unsigned char)s[i])) ++i;
            if (i>=n) break;
            size_t j=i;
            while (j<n && !isspace((unsigned char)s[j])) ++j;
            std::string w = s.substr(i,j-i);
            auto it = vocab.find(w);
            if (it==vocab.end()) it = vocab.emplace(w,nextId++).first;
            ids.push_back(it->second);
            i=j;
        }
        return ids;
    };

    // 3) Read file lines
    std::vector<std::vector<int>> lines;
    lines.reserve(100000);
    FILE* f = fopen(filepath,"r");
    if (!f) { perror("fopen"); return 1; }
    char buf[MAX_INPUT_SIZE];
    while (lines.size()<MAX_LINES && fgets(buf,sizeof buf,f)) {
        std::string s(buf);
        normalize(s);
        lines.push_back(tokenizeIds(s));
    }
    fclose(f);
    int lineCount = lines.size();
    if (lineCount==0) { printf("No lines read.\n"); return 0; }

    // 4) Parse & tokenize phrases (keep original strings to print)
    std::vector<std::string> phraseStrs;
    std::vector<std::vector<int>> phrases;
    {
        char *saveptr=nullptr;
        for (char* tok=strtok(phrase_line,"," ); tok && phrases.size()<MAX_PHRASES;
             tok=strtok(nullptr,"," )) {
            while(*tok==' ') ++tok;
            char* end=tok+strlen(tok)-1;
            while(end>tok&&*end==' ') *end--='\0';
            if (!*tok) continue;
            std::string ps(tok);
            normalize(ps);
            auto ids = tokenizeIds(ps);
            if (!ids.empty()) {
                phraseStrs.push_back(std::string(tok));
                phrases.push_back(ids);
            }
        }
    }
    int phraseCount = phrases.size();
    if (phraseCount==0) { printf("No phrases.\n"); return 0; }

    // 5) Determine max token lengths
    int maxLineTokens=0;
    std::vector<int> lineLens(lineCount);
    for (int i=0;i<lineCount;++i) {
        int L = lines[i].size();
        lineLens[i]=L;
        if (L>maxLineTokens) maxLineTokens=L;
    }
    if (maxLineTokens>MAX_TOKENS_PER_LINE) maxLineTokens=MAX_TOKENS_PER_LINE;

    int maxPhraseTokens=0;
    std::vector<int> phraseLens(phraseCount);
    for (int i=0;i<phraseCount;++i) {
        int L=phrases[i].size();
        phraseLens[i]=L;
        if (L>maxPhraseTokens) maxPhraseTokens=L;
    }
    if (maxPhraseTokens>MAX_TOKENS_PER_PHRASE) maxPhraseTokens=MAX_TOKENS_PER_PHRASE;

    // 6) Pack into flat host arrays (pad with 0)
    size_t lb = size_t(lineCount)*maxLineTokens;
    size_t pb = size_t(phraseCount)*maxPhraseTokens;
    int *h_lineTokens   = (int*)calloc(lb, sizeof(int));
    int *h_phraseTokens = (int*)calloc(pb, sizeof(int));
    for (int i=0;i<lineCount;++i)
      for (int j=0;j<lineLens[i] && j<maxLineTokens;++j)
        h_lineTokens[i*maxLineTokens + j] = lines[i][j];
    for (int i=0;i<phraseCount;++i)
      for (int j=0;j<phraseLens[i] && j<maxPhraseTokens;++j)
        h_phraseTokens[i*maxPhraseTokens + j] = phrases[i][j];

    // 7) Copy constants & buffers to device
    cudaMemcpyToSymbol(d_lineCount,      &lineCount,      sizeof(int));
    cudaMemcpyToSymbol(d_phraseCount,    &phraseCount,    sizeof(int));
    cudaMemcpyToSymbol(d_maxLineTokens,  &maxLineTokens,  sizeof(int));
    cudaMemcpyToSymbol(d_maxPhraseTokens,&maxPhraseTokens,sizeof(int));

    int *d_lineTokens,*d_lineLens;
    int *d_phraseTokens,*d_phraseLens;
    int *d_counts;
    size_t lb_b = lb*sizeof(int),
           pb_b = pb*sizeof(int),
           ll_b = lineCount*sizeof(int),
           pl_b = phraseCount*sizeof(int),
           ct_b = phraseCount*sizeof(int);

    cudaMalloc(&d_lineTokens, lb_b);
    cudaMalloc(&d_lineLens,   ll_b);
    cudaMalloc(&d_phraseTokens,pb_b);
    cudaMalloc(&d_phraseLens, pl_b);
    cudaMalloc(&d_counts,     ct_b);

    cudaMemcpy(d_lineTokens,   h_lineTokens, lb_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lineLens,     lineLens.data(), ll_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_phraseTokens, h_phraseTokens, pb_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_phraseLens,   phraseLens.data(), pl_b, cudaMemcpyHostToDevice);
    cudaMemset(d_counts, 0, ct_b);

    // 8) Determine threads/block & blocks, print it
    int tpb = (argc>1?atoi(argv[1]):256);
    if (tpb<1||tpb>1024) tpb = 256;
    int blocks = (lineCount + tpb - 1) / tpb;
    printf("Using %d CUDA threads per block and %d blocks\n",
           tpb, blocks);

    // 9) Launch kernel & time it
    double t0 = clock() / (double)CLOCKS_PER_SEC;
    tokenSearchKernel<<<blocks,tpb>>>(
        d_lineTokens, d_lineLens,
        d_phraseTokens, d_phraseLens,
        d_counts
    );
    cudaDeviceSynchronize();
    double t1 = clock() / (double)CLOCKS_PER_SEC;

    // 10) Copy back counts & print table
    int *h_counts = (int*)malloc(ct_b);
    cudaMemcpy(h_counts, d_counts, ct_b, cudaMemcpyDeviceToHost);

    printf("\n+-------------------------------+---------------+\n"
           "| %-29s | %13s |\n"
           "+-------------------------------+---------------+\n",
           "Phrase","Matches");

    int total = 0;
    for (int i = 0; i < phraseCount; ++i) {
        printf("| %-29s | %13d |\n",
               phraseStrs[i].c_str(), h_counts[i]);
        total += h_counts[i];
    }
    printf("+-------------------------------+---------------+\n"
           "| %-29s | %13d |\n"
           "+-------------------------------+---------------+\n"
           "| %-29s | %13.4f |\n"
           "+-------------------------------+---------------+\n",
           "Total matches", total,
           "Elapsed time (s)", t1 - t0);

    // 11) CSV logging (omitted for brevity)...

    // 12) Cleanup
    cudaFree(d_lineTokens);
    cudaFree(d_lineLens);
    cudaFree(d_phraseTokens);
    cudaFree(d_phraseLens);
    cudaFree(d_counts);
    free(h_lineTokens);
    free(h_phraseTokens);
    free(h_counts);

    return 0;
}
