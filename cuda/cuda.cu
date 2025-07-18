// cuda.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <ctime>
#include <cuda_runtime.h>

#define MAX_LINE_LENGTH     8192
#define MAX_PHRASES         100
#define MAX_WORDS_PER_LINE  512
#define MAX_LINES           200000
#define MAX_INPUT_SIZE      4096
#define CSV_FILENAME        "results.csv"

// ——— serial normalize: drop punctuation, lowercase —————————————————
void normalize(char* s) {
    char* dst = s;
    for (char* src = s; *src; ++src) {
        unsigned char c = (unsigned char)*src;
        if (isalpha(c) || isspace(c))
            *dst++ = tolower(c);
    }
    *dst = '\0';
}

// ——— serial tokenize ——————————————————————————————————————————
int tokenize(char* line, char* words[], int maxw) {
    int n = 0;
    char* tok = strtok(line, " \t\r\n");
    while (tok && n < maxw) {
        words[n++] = tok;
        tok = strtok(NULL, " \t\r\n");
    }
    return n;
}

// ——— serial phrase_match ——————————————————————————————————————
int phrase_match(char* words[], int wc, const char* phrase) {
    char buf[MAX_LINE_LENGTH];
    strncpy(buf, phrase, sizeof buf);
    buf[sizeof buf-1] = '\0';
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

// ——— serial search_and_log ————————————————————————————————————
void search_and_log(const char* filename, char* phrases[], int pc) {
    FILE* f = fopen(filename, "r");
    if (!f) { perror("Error opening file"); return; }

    int counts[MAX_PHRASES] = {0}, total = 0;
    char line[MAX_LINE_LENGTH];
    clock_t t0 = clock();

    while (fgets(line, sizeof line, f)) {
        normalize(line);
        char* words[MAX_WORDS_PER_LINE];
        int wc = tokenize(line, words, MAX_WORDS_PER_LINE);
        for (int i = 0; i < pc; ++i) {
            if (phrase_match(words, wc, phrases[i])) {
                ++counts[i];
                ++total;
            }
        }
    }
    fclose(f);
    double secs = double(clock() - t0) / CLOCKS_PER_SEC;

    // print table
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

    // append to CSV
    static int header = 0;
    FILE* csv = fopen(CSV_FILENAME, "a");
    if (!csv) { perror("Could not open results.csv"); return; }
    if (!header) {
        fprintf(csv, "timestamp,filename,phrases,total_matches,time_s\n");
        header = 1;
    }
    char plist[MAX_INPUT_SIZE] = "";
    for (int i = 0; i < pc; ++i) {
        if (i) strcat(plist, ";");
        strcat(plist, phrases[i]);
    }
    time_t now = time(NULL);
    char tbuf[64];
    strftime(tbuf, sizeof tbuf, "%Y-%m-%d %H:%M:%S", localtime(&now));
    fprintf(csv, "\"%s\",\"%s\",\"%s\",%d,%.4f\n",
            tbuf, filename, plist, total, secs);
    fclose(csv);
}

int main(int argc, char** argv) {
    // --- New: allow user to override CUDA threads-per-block ---
    int threadsPerBlock = 256;
    if (argc > 1) {
        int t = atoi(argv[1]);
        if (t > 0) threadsPerBlock = t;
    }
    // (in this serial-only version we don't actually launch any kernels,
    // but if you do, you'd use threadsPerBlock in <<<>>>)

    char filepath[MAX_INPUT_SIZE], phrase_line[MAX_INPUT_SIZE];
    char* phrases[MAX_PHRASES];
    int pc = 0;

    // read inputs
    printf("Enter path to text file: ");
    if (!fgets(filepath, sizeof filepath, stdin)) return 0;
    filepath[strcspn(filepath, "\r\n")] = '\0';

    printf("Enter search phrases, comma-separated:\n");
    if (!fgets(phrase_line, sizeof phrase_line, stdin)) return 0;
    phrase_line[strcspn(phrase_line, "\r\n")] = '\0';

    // split phrases
    for (char* tok = strtok(phrase_line, ","); tok && pc < MAX_PHRASES; tok = strtok(NULL, ",")) {
        while (*tok == ' ') ++tok;
        char* end = tok + strlen(tok) - 1;
        while (end > tok && *end == ' ') *end-- = '\0';
        if (*tok) phrases[pc++] = strdup(tok);
    }
    if (pc == 0) return 0;

    // New printf:
    printf("Using %d threads per block. Starting search...\n", threadsPerBlock);

    // perform the search
    search_and_log(filepath, phrases, pc);
    return 0;
}
