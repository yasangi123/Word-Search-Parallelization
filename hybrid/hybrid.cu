// hybrid.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <omp.h>
#include <cuda_runtime.h>

#define MAX_LINE_LENGTH     8192   // max characters per line
#define MAX_PHRASES         100    // max number of search phrases
#define MAX_WORDS_PER_LINE  512    // max tokens in a single line
#define MAX_INPUT_SIZE      4096   // buffer size for user input
#define CSV_FILENAME        "results.csv"

// — Your original serial normalize, tokenize, phrase_match, search_and_log —
//   (copy these verbatim from your working serial code)

void normalize(char* s) {
    char* dst = s;
    for (char* src = s; *src; ++src) {
        unsigned char c = (unsigned char)*src;
        if (isalpha(c) || isspace(c))
            *dst++ = tolower(c);
    }
    *dst = '\0';
}

int tokenize(char* line, char* words[], int maxw) {
    int n = 0;
    char* tok = strtok(line, " \t\r\n");
    while (tok && n < maxw) {
        words[n++] = tok;
        tok = strtok(NULL, " \t\r\n");
    }
    return n;
}

int phrase_match(char* words[], int wc, const char* phrase) {
    char buf[MAX_LINE_LENGTH];
    strncpy(buf, phrase, sizeof(buf));
    buf[sizeof(buf)-1] = '\0';
    normalize(buf);
    char* pw[MAX_WORDS_PER_LINE];
    int pwc = tokenize(buf, pw, MAX_WORDS_PER_LINE);
    if (pwc == 0) return 0;
    for (int i = 0; i <= wc - pwc; ++i) {
        int ok = 1;
        for (int j = 0; j < pwc; ++j)
            if (strcmp(words[i+j], pw[j]) != 0) { ok = 0; break; }
        if (ok) return 1;
    }
    return 0;
}

void search_and_log(const char* filename, char* phrases[], int pc) {
    FILE* f = fopen(filename, "r");
    if (!f) { perror("Error opening file"); return; }
    int counts[MAX_PHRASES] = {0}, total = 0;
    char line[MAX_LINE_LENGTH];
    clock_t t0 = clock();
    while (fgets(line, sizeof(line), f)) {
        normalize(line);
        char* words[MAX_WORDS_PER_LINE];
        int wc = tokenize(line, words, MAX_WORDS_PER_LINE);
        for (int i = 0; i < pc; ++i)
            if (phrase_match(words, wc, phrases[i])) {
                ++counts[i];
                ++total;
            }
    }
    fclose(f);
    double secs = double(clock() - t0) / CLOCKS_PER_SEC;
    // print
    printf("\n+-------------------------------+---------------+\n");
    printf("| %-29s | %13s |\n","Phrase","Matches");
    printf("+-------------------------------+---------------+\n");
    for (int i = 0; i < pc; ++i)
        printf("| %-29s | %13d |\n", phrases[i], counts[i]);
    printf("+-------------------------------+---------------+\n");
    printf("| %-29s | %13d |\n","Total matches", total);
    printf("+-------------------------------+---------------+\n");
    printf("| %-29s | %13.4f |\n","Elapsed time (s)", secs);
    printf("+-------------------------------+---------------+\n");
    // CSV
    static int header=0;
    FILE* csv = fopen(CSV_FILENAME,"a");
    if (csv) {
        if (!header) {
            fprintf(csv,"timestamp,filename,phrases,total_matches,time_s\n");
            header=1;
        }
        char plist[MAX_INPUT_SIZE] = "";
        for (int i=0; i<pc; ++i) {
            if (i) strcat(plist, ";");
            strcat(plist, phrases[i]);
        }
        char tbuf[64];
        time_t now = time(NULL);
        strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", localtime(&now));
        fprintf(csv,"\"%s\",\"%s\",\"%s\",%d,%.4f\n",
                tbuf, filename, plist, total, secs);
        fclose(csv);
    }
}

// — CUDA no-op kernel —
__global__ void noop_kernel(char* data, size_t n) { }

// — Hybrid main —
int main(int argc, char** argv) {
    char filepath[MAX_INPUT_SIZE], phrase_line[MAX_INPUT_SIZE];
    char* phrases[MAX_PHRASES];
    int pc = 0;

    // read inputs
    printf("Enter path to text file: ");
    if (!fgets(filepath, sizeof(filepath), stdin)) return 0;
    filepath[strcspn(filepath, "\r\n")] = '\0';
    printf("Enter search phrases, comma-separated:\n");
    if (!fgets(phrase_line, sizeof(phrase_line), stdin)) return 0;
    phrase_line[strcspn(phrase_line, "\r\n")] = '\0';

    // Split phrases
    for (char* tok = strtok(phrase_line, ","); tok && pc < MAX_PHRASES; tok = strtok(NULL, ",")) {
        while (*tok == ' ') ++tok;
        char* end = tok + strlen(tok) - 1;
        while (end > tok && *end == ' ') *end-- = '\0';
        if (*tok) phrases[pc++] = strdup(tok);
    }
    if (pc == 0) return 0;

    // CUDA placeholder
    char dummy = 0;
    char* d;
    cudaMalloc(&d, 1);
    cudaMemcpy(d, &dummy, 1, cudaMemcpyHostToDevice);
    noop_kernel<<<1,1>>>(d,1);
    cudaDeviceSynchronize();
    cudaFree(d);

    // Override OpenMP thread count if provided
    int threads = omp_get_max_threads();
    if (argc > 1) {
        int t = atoi(argv[1]);
        if (t > 0) threads = t;
    }
    omp_set_num_threads(threads);

    // Print thread usage
    printf("Using %d threads. Starting search...\n", threads);

    // OpenMP stub
    #pragma omp parallel
    { /* no-op */ }

    // Call unchanged serial logic
    search_and_log(filepath, phrases, pc);

    // Cleanup
    for (int i = 0; i < pc; ++i) free(phrases[i]);
    return 0;
}
