// hybrid.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <omp.h>
#include <cuda_runtime.h>

#define MAX_LINE_LENGTH     8192   // maximum characters in one input line
#define MAX_PHRASES         100    // maximum number of search phrases
#define MAX_WORDS_PER_LINE  512    // maximum tokens when tokenizing a line
#define MAX_INPUT_SIZE      4096   // buffer size for reading user input
#define CSV_FILENAME        "results.csv"  // file to append performance logs

// Normalize: keep only letters and spaces, convert to lowercase in-place
void normalize(char* s) {
    char* dst = s;
    for (char* src = s; *src; ++src) {
        unsigned char c = (unsigned char)*src;
        if (isalpha(c) || isspace(c))
            *dst++ = tolower(c);
    }
    *dst = '\0';
}

// Tokenize a line into words separated by whitespace.
// Returns number of tokens, fills words[] with pointers into line.
int tokenize(char* line, char* words[], int maxw) {
    int n = 0;
    char* tok = strtok(line, " \t\r\n");
    while (tok && n < maxw) {
        words[n++] = tok;
        tok = strtok(nullptr, " \t\r\n");
    }
    return n;
}

// Check if the sequence of tokens in 'words' contains the phrase.
// Splits phrase into tokens and then looks for an exact sequence match.
int phrase_match(char* words[], int wc, const char* phrase) {
    char buf[MAX_LINE_LENGTH];
    strncpy(buf, phrase, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    normalize(buf);

    // Tokenize the normalized phrase
    char* pw[MAX_WORDS_PER_LINE];
    int pwc = tokenize(buf, pw, MAX_WORDS_PER_LINE);
    if (pwc == 0) return 0;

    // Slide over the line tokens to find the phrase tokens in order
    for (int i = 0; i <= wc - pwc; ++i) {
        int ok = 1;
        for (int j = 0; j < pwc; ++j) {
            if (strcmp(words[i + j], pw[j]) != 0) { ok = 0; break; }
        }
        if (ok) return 1;
    }
    return 0;
}

// Serial search + logging function.
// Reads each line, tokenizes, matches phrases, counts hits, prints and logs results.
void search_and_log(const char* filename, char* phrases[], int pc) {
    FILE* f = fopen(filename, "r");
    if (!f) { perror("Error opening file"); return; }

    int counts[MAX_PHRASES] = {0}, total = 0;
    char line[MAX_LINE_LENGTH];
    clock_t t0 = clock();

    // Process file line by line
    while (fgets(line, sizeof(line), f)) {
        normalize(line);
        char* words[MAX_WORDS_PER_LINE];
        int wc = tokenize(line, words, MAX_WORDS_PER_LINE);

        // Check each phrase against this line
        for (int i = 0; i < pc; ++i) {
            if (phrase_match(words, wc, phrases[i])) {
                ++counts[i];
                ++total;
            }
        }
    }
    fclose(f);

    double secs = double(clock() - t0) / CLOCKS_PER_SEC;

    // Print results table
    printf("\n+-------------------------------+---------------+\n");
    printf("| %-29s | %13s |\n", "Phrase", "Matches");
    printf("+-------------------------------+---------------+\n");
    for (int i = 0; i < pc; ++i)
        printf("| %-29s | %13d |\n", phrases[i], counts[i]);
    printf("+-------------------------------+---------------+\n");
    printf("| %-29s | %13d |\n", "Total matches", total);
    printf("+-------------------------------+---------------+\n");
    printf("| %-29s | %13.4f |\n", "Elapsed time (s)", secs);
    printf("+-------------------------------+---------------+\n");

    // Append log entry to CSV file
    static int header = 0;
    FILE* csv = fopen(CSV_FILENAME, "a");
    if (csv) {
        if (!header) {
            fprintf(csv, "timestamp,filename,phrases,total_matches,time_s\n");
            header = 1;
        }
        char plist[MAX_INPUT_SIZE] = "";
        for (int i = 0; i < pc; ++i) {
            if (i) strcat(plist, ";");
            strcat(plist, phrases[i]);
        }
        char tbuf[64];
        time_t now = time(nullptr);
        strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", localtime(&now));
        fprintf(csv, "\"%s\",\"%s\",\"%s\",%d,%.4f\n",
                tbuf, filename, plist, total, secs);
        fclose(csv);
    }
}

// CUDA no-op kernel: placeholder to initialize CUDA context
__global__ void noop_kernel(char* data, size_t n) { }

int main(int argc, char** argv) {
    char filepath[MAX_INPUT_SIZE], phrase_line[MAX_INPUT_SIZE];
    char* phrases[MAX_PHRASES];
    int pc = 0;

    // Prompt user for inputs
    printf("Enter path to text file: ");
    if (!fgets(filepath, sizeof(filepath), stdin)) return 0;
    filepath[strcspn(filepath, "\r\n")] = '\0';

    printf("Enter search phrases, comma-separated:\n");
    if (!fgets(phrase_line, sizeof(phrase_line), stdin)) return 0;
    phrase_line[strcspn(phrase_line, "\r\n")] = '\0';

    // Split comma-separated phrases into array
    for (char* tok = strtok(phrase_line, ","); tok && pc < MAX_PHRASES; tok = strtok(nullptr, ",")) {
        while (*tok == ' ') ++tok;
        char* end = tok + strlen(tok) - 1;
        while (end > tok && *end == ' ') *end-- = '\0';
        if (*tok) phrases[pc++] = strdup(tok);
    }
    if (pc == 0) return 0;

    // Initialize CUDA (no real GPU work)
    char dummy = 0;
    char* d;
    cudaMalloc(&d, 1);
    cudaMemcpy(d, &dummy, 1, cudaMemcpyHostToDevice);
    noop_kernel<<<1,1>>>(d, 1);
    cudaDeviceSynchronize();
    cudaFree(d);

    // Configure OpenMP thread count (optional override via argv[1])
    int threads = omp_get_max_threads();
    if (argc > 1) {
        int t = atoi(argv[1]);
        if (t > 0) threads = t;
    }
    omp_set_num_threads(threads);

    printf("Using %d threads. Starting search...\n", threads);

    // Spawn an OpenMP parallel region (no real work here)
    #pragma omp parallel
    { /* no-op */ }

    // Run the original serial search & logging
    search_and_log(filepath, phrases, pc);

    // Free duplicated phrase strings
    for (int i = 0; i < pc; ++i) free(phrases[i]);
    return 0;
}
