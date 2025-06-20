#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <omp.h>

#define MAX_LINE_LENGTH 8192      // Max characters per line
#define MAX_PHRASES 100           // Max number of phrases to search
#define MAX_WORDS_PER_LINE 512    // Max tokens per line
#define MAX_INPUT_SIZE 4096       // Buffer for user input
#define NUM_THREADS 8             

// ——— clean up text ————————————————————————————————————————
// Strips punctuation and lowercases letters, in-place.
void normalize(char *s) {
    char *dst = s;
    for (char *src = s; *src; ++src) {
        unsigned char c = (unsigned char)*src;
        // keep only letters and spaces, in lowercase
        if (isalpha(c) || isspace(c))
            *dst++ = tolower(c);
    }
    *dst = '\0';
}

// ——— split a line into words —————————————————————————————
// Thread-safe tokenization using strtok_r. Returns the number of words.
int tokenize(char *line, char *words[], int maxw) {
    int n = 0;
    char *saveptr;
    char *tok = strtok_r(line, " \t\r\n", &saveptr);
    while (tok && n < maxw) {
        words[n++] = tok;
        tok = strtok_r(NULL, " \t\r\n", &saveptr);
    }
    return n;
}

// ——— Check if a phrase appears as consecutive words ———————————————————
// Copies, normalizes, tokenizes the phrase, then scans the line tokens.
int phrase_match(char *words[], int wc, const char *phrase) {
    char buf[MAX_LINE_LENGTH];
    // copy phrase into modifiable buffer
    strncpy(buf, phrase, sizeof buf);
    buf[sizeof buf - 1] = '\0';

    normalize(buf);  // strip punctuation + lowercase

    char *pw[MAX_WORDS_PER_LINE];
    int pwc = tokenize(buf, pw, MAX_WORDS_PER_LINE);
    if (pwc == 0) return 0;  // empty after cleaning

    // sliding-window compare
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

// ——— Parallel search routine —————————————————————————————————————
// Loads the entire file into memory, then divides the work among threads.
void search_and_log_openmp(const char *filename, char *phrases[], int pc) {
    // 1) Read all lines into an expandable array
    FILE *f = fopen(filename, "r");
    if (!f) { perror("Error opening file"); return; }
    int cap = 1024, line_count = 0;
    char **lines = malloc(cap * sizeof *lines);
    char buf[MAX_LINE_LENGTH];
    while (fgets(buf, sizeof buf, f)) {
        if (line_count >= cap) {
            cap *= 2;
            lines = realloc(lines, cap * sizeof *lines);
            if (!lines) { fprintf(stderr, "Out of memory\n"); exit(1); }
        }
        lines[line_count++] = strdup(buf);
    }
    fclose(f);

    int counts[MAX_PHRASES] = {0}, total = 0;

    // 2) Parallel region: each thread processes a chunk of lines
    omp_set_num_threads(NUM_THREADS);
    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        int lc[MAX_PHRASES] = {0}, lt = 0;  // local counters

        #pragma omp for schedule(static)
        for (int i = 0; i < line_count; ++i) {
            // make a private copy before modifying
            char line_copy[MAX_LINE_LENGTH];
            strncpy(line_copy, lines[i], sizeof line_copy);
            line_copy[sizeof line_copy - 1] = '\0';

            normalize(line_copy);

            // split into words
            char *words[MAX_WORDS_PER_LINE];
            int wc = tokenize(line_copy, words, MAX_WORDS_PER_LINE);

            // test each phrase
            for (int j = 0; j < pc; ++j) {
                if (phrase_match(words, wc, phrases[j])) {
                    lc[j]++; 
                    lt++;
                }
            }
        }

        // safely merge each thread's counts into the global totals
        #pragma omp critical
        {
            for (int j = 0; j < pc; ++j) 
                counts[j] += lc[j];
            total += lt;
        }
    }

    double t1 = omp_get_wtime();

    // 3) Print results in a table
    printf("\n+-------------------------------+------------+\n");
    printf("| %-29s | %10s |\n", "Phrase", "Matches");
    printf("+-------------------------------+------------+\n");
    for (int i = 0; i < pc; ++i)
        printf("| %-29s | %10d |\n", phrases[i], counts[i]);
    printf("+-------------------------------+------------+\n");
    printf("| %-29s | %10d |\n", "Total matches", total);
    printf("+-------------------------------+------------+\n");
    printf("| %-29s | %10.4f |\n", "Elapsed time (s)", t1 - t0);
    printf("+-------------------------------+------------+\n");

    // 4) Append a timestamped row to results.csv
    {
        static int header = 0;
        FILE *csv = fopen("results.csv", "a");
        if (csv) {
            if (!header) {
                fprintf(csv,
                    "timestamp,filename,phrases,total_matches,time_s\n");
                header = 1;
            }
            // join phrases with ';'
            char plist[MAX_INPUT_SIZE] = "";
            for (int i = 0; i < pc; ++i) {
                if (i) strcat(plist, ";");
                strcat(plist, phrases[i]);
            }
            // timestamp
            char ts[64];
            time_t now = time(NULL);
            strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", localtime(&now));

            fprintf(csv, "\"%s\",\"%s\",\"%s\",%d,%.4f\n",
                ts, filename, plist, total, t1 - t0);
            fclose(csv);
        }
    }

    // 5) Cleanup allocated memory
    for (int i = 0; i < line_count; ++i)
        free(lines[i]);
    free(lines);
}

// ——— Main: user interface ————————————————————————————————————————

int main(void) {
    char filepath[MAX_INPUT_SIZE], phrase_line[MAX_INPUT_SIZE];
    char *phrases[MAX_PHRASES];
    int pc = 0;

    // Ask for the text file path
    printf("Enter path to text file: ");
    if (!fgets(filepath, sizeof filepath, stdin))
        return 0;
    filepath[strcspn(filepath, "\r\n")] = '\0';  // drop the newline

    // Ask for comma-separated phrases
    printf("Enter search phrases, comma-separated:\n");
    if (!fgets(phrase_line, sizeof phrase_line, stdin))
        return 0;
    phrase_line[strcspn(phrase_line, "\r\n")] = '\0';

    // Split the phrase line at commas (thread-safe)
    char *saveptr, *tok = strtok_r(phrase_line, ",", &saveptr);
    while (tok && pc < MAX_PHRASES) {
        // trim leading spaces
        while (*tok == ' ') ++tok;
        // trim trailing spaces
        char *end = tok + strlen(tok) - 1;
        while (end > tok && *end == ' ') *end-- = '\0';

        if (*tok)  // if non-empty
            phrases[pc++] = tok;
        tok = strtok_r(NULL, ",", &saveptr);
    }

    if (pc == 0) {
        printf("No valid phrases entered. Exiting.\n");
        return 0;
    }

    // Show how many threads we'll use, then run
    printf("Using %d threads. Starting search...\n", NUM_THREADS);
    search_and_log_openmp(filepath, phrases, pc);
    return 0;
}
