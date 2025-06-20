#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

#define MAX_LINE_LENGTH     8192   // max characters per line
#define MAX_PHRASES         100    // max number of search phrases
#define MAX_WORDS_PER_LINE  512    // max tokens in a single line
#define MAX_INPUT_SIZE      4096   // buffer for user input

// ——— clean up a string in-place ———————————————————————————————
// Removes punctuation, lowercases letters, keeps spaces.
void normalize(char* s) {
    char* dst = s;
    for (char* src = s; *src; ++src) {
        if (isalpha((unsigned char)*src) || isspace((unsigned char)*src))
            *dst++ = tolower((unsigned char)*src);
    }
    *dst = '\0';
}

// ——— split a string into words ———————————————————————————————
// Modifies 'line' by inserting '\0' at delimiters, returns word count.
int tokenize(char* line, char* words[], int maxw) {
    int n = 0;
    char* tok = strtok(line, " \t\r\n");
    while (tok && n < maxw) {
        words[n++] = tok;
        tok = strtok(NULL, " \t\r\n");
    }
    return n;
}

// ——— Check if a multi-word phrase exists in an array of tokens —————————
// Copies phrase into buf, normalizes it, tokenizes it, then
// does a sliding-window compare against 'words[]'.
int phrase_match(char* words[], int wc, const char* phrase) {
    char buf[MAX_LINE_LENGTH];
    strncpy(buf, phrase, sizeof buf);
    buf[sizeof buf - 1] = '\0';
    normalize(buf);

    char* pw[MAX_WORDS_PER_LINE];
    int pwc = tokenize(buf, pw, MAX_WORDS_PER_LINE);
    if (pwc == 0) return 0;  // empty phrase after cleaning

    // slide through the line’s words
    for (int i = 0; i <= wc - pwc; ++i) {
        int match = 1;
        for (int j = 0; j < pwc; ++j) {
            if (strcmp(words[i + j], pw[j]) != 0) {
                match = 0;
                break;
            }
        }
        if (match) return 1;
    }
    return 0;
}

// ——— Core: read file, search phrases, and log results ——————————————
void search_and_log(const char* filename, char* phrases[], int pc) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        perror("Error opening file");
        return;
    }

    int counts[MAX_PHRASES] = {0}, total = 0;
    char line[MAX_LINE_LENGTH];

    // start timer
    clock_t t0 = clock();
    while (fgets(line, sizeof line, f)) {
        normalize(line);

        // break the cleaned line into words
        char* words[MAX_WORDS_PER_LINE];
        int wc = tokenize(line, words, MAX_WORDS_PER_LINE);

        // check each phrase against this line
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

    // ——— display results in a friendly table —————————————————————
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

    // ——— append this run’s data to a CSV for later analysis ————————
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
        // join phrases with semicolons
        char plist[MAX_INPUT_SIZE] = "";
        for (int i = 0; i < pc; ++i) {
            if (i) strcat(plist, ";");
            strcat(plist, phrases[i]);
        }
        // create timestamp
        time_t now = time(NULL);
        char tbuf[64];
        strftime(tbuf, sizeof tbuf, "%Y-%m-%d %H:%M:%S", localtime(&now));

        fprintf(csv, "\"%s\",\"%s\",\"%s\",%d,%.4f\n",
                tbuf, filename, plist, total, secs);
        fclose(csv);
    }
}

// ——— Main: interactive menu to get inputs ——————————————————————
int main(void) {
    char filepath[MAX_INPUT_SIZE];
    char phrase_line[MAX_INPUT_SIZE];
    char* phrases[MAX_PHRASES];
    int phrase_count = 0;

    // ask for the file to search
    printf("Enter path to text file: ");
    if (!fgets(filepath, sizeof filepath, stdin)) return 0;
    filepath[strcspn(filepath, "\r\n")] = '\0';  // remove newline

    // ask for comma-separated phrases
    printf("Enter search phrases, comma-separated:\n");
    if (!fgets(phrase_line, sizeof phrase_line, stdin)) return 0;
    phrase_line[strcspn(phrase_line, "\r\n")] = '\0';

    // split the phrase list by commas, trimming spaces
    char* tok = strtok(phrase_line, ",");
    while (tok && phrase_count < MAX_PHRASES) {
        while (*tok == ' ') ++tok;                // trim leading
        char* end = tok + strlen(tok) - 1;
        while (end > tok && *end == ' ') *end-- = '\0';  // trim trailing
        if (*tok) phrases[phrase_count++] = tok;
        tok = strtok(NULL, ",");
    }
    if (phrase_count == 0) {
        printf("No valid phrases entered. Exiting.\n");
        return 0;
    }

    // perform the search and record results
    search_and_log(filepath, phrases, phrase_count);
    return 0;
}
