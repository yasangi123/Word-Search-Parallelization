#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

#define MAX_LINE_LENGTH     8192
#define MAX_PHRASES         100
#define MAX_WORDS_PER_LINE  512
#define MAX_INPUT_SIZE      4096

// ——— Utility Functions —————————————————————————————————————————

// Lowercase in-place
void to_lowercase(char* s) {
    for (char* p = s; *p; ++p)
        *p = tolower((unsigned char)*p);
}

// Remove punctuation & lowercase
void normalize(char* s) {
    char* dst = s;
    for (char* src = s; *src; ++src) {
        if (isalpha((unsigned char)*src) || isspace((unsigned char)*src))
            *dst++ = tolower((unsigned char)*src);
    }
    *dst = '\0';
}

// Tokenize line into words; returns count
int tokenize(char* line, char* words[], int maxw) {
    int n = 0;
    char* tok = strtok(line, " \t\r\n");
    while (tok && n < maxw) {
        words[n++] = tok;
        tok = strtok(NULL, " \t\r\n");
    }
    return n;
}

// Return 1 if phrase (one or more words) found in token array
int phrase_match(char* words[], int wc, const char* phrase) {
    char buf[MAX_LINE_LENGTH];
    strncpy(buf, phrase, MAX_LINE_LENGTH);
    normalize(buf);
    char* pw[MAX_WORDS_PER_LINE];
    int pwc = tokenize(buf, pw, MAX_WORDS_PER_LINE);
    if (pwc == 0) return 0;
    for (int i = 0; i <= wc - pwc; ++i) {
        int ok = 1;
        for (int j = 0; j < pwc; ++j) {
            if (strcmp(words[i + j], pw[j]) != 0) {
                ok = 0;
                break;
            }
        }
        if (ok) return 1;
    }
    return 0;
}

// ——— Core Search & Logging ——————————————————————————————————————

void search_and_log(const char* filename, char* phrases[], int pc) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        perror("Error opening file");
        return;
    }

    int counts[MAX_PHRASES] = {0};
    int total = 0;
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
    clock_t t1 = clock();
    fclose(f);

    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;

    // ——— Neat ASCII Table Output —————————————————————————————
    printf("\n+-------------------------------+---------------+\n");
    printf("| %-29s | %13s |\n", "Phrase", "Matches");
    printf("+-------------------------------+---------------+\n");
    for (int i = 0; i < pc; ++i) {
        printf("| %-29s | %13d |\n", phrases[i], counts[i]);
    }
    printf("+-------------------------------+---------------+\n");
    printf("| %-29s | %13d |\n", "Total matches", total);
    printf("+-------------------------------+---------------+\n");
    printf("| %-29s | %13.4f |\n", "Elapsed time (s)", secs);
    printf("+-------------------------------+---------------+\n");

    // ——— Append to CSV Log —————————————————————————————————————
    {
        static int header_done = 0;
        FILE* csv = fopen("results.csv", "a");
        if (!csv) {
            perror("Could not open results.csv");
            return;
        }
        if (!header_done) {
            fprintf(csv, "timestamp,filename,phrases,total_matches,time_s\n");
            header_done = 1;
        }
        // Build semicolon-separated phrase list
        char plist[MAX_INPUT_SIZE] = "";
        for (int i = 0; i < pc; ++i) {
            if (i) strcat(plist, ";");
            strcat(plist, phrases[i]);
        }
        // Timestamp
        time_t now = time(NULL);
        struct tm* tm = localtime(&now);
        char tbuf[64];
        strftime(tbuf, sizeof tbuf, "%Y-%m-%d %H:%M:%S", tm);

        fprintf(csv, "\"%s\",\"%s\",\"%s\",%d,%.4f\n",
                tbuf, filename, plist, total, secs);
        fclose(csv);
    }
}

// ——— Simple Menu UI —————————————————————————————————————————

int main(void) {
    char filepath[MAX_INPUT_SIZE];
    char phrase_line[MAX_INPUT_SIZE];
    char* phrases[MAX_PHRASES];
    int phrase_count = 0;

    // 1) Get file path
    printf("Enter path to text file: ");
    if (!fgets(filepath, sizeof filepath, stdin)) return 0;
    filepath[strcspn(filepath, "\r\n")] = 0;

    // 2) Get comma-separated phrases
    printf("Enter search phrases, comma-separated:\n");
    if (!fgets(phrase_line, sizeof phrase_line, stdin)) return 0;
    phrase_line[strcspn(phrase_line, "\r\n")] = 0;

    // 3) Split into phrases[]
    char* tok = strtok(phrase_line, ",");
    while (tok && phrase_count < MAX_PHRASES) {
        // Trim leading/trailing spaces
        while (*tok == ' ') ++tok;
        char* end = tok + strlen(tok) - 1;
        while (end > tok && *end == ' ') *end-- = '\0';
        if (*tok) phrases[phrase_count++] = tok;
        tok = strtok(NULL, ",");
    }
    if (phrase_count == 0) {
        printf("No valid phrases entered. Exiting..\n");
        return 0;
    }

    // 4) Execute search and log results
    search_and_log(filepath, phrases, phrase_count);

    return 0;
}
