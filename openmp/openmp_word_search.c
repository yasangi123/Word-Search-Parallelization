#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <omp.h>

#define MAX_LINE_LENGTH     8192
#define MAX_PHRASES         100
#define MAX_WORDS_PER_LINE  512
#define MAX_INPUT_SIZE      4096

// ——— Utility Functions —————————————————————————————————————————

// Remove punctuation & lowercase (in-place)
void normalize(char* s) {
    char* dst = s;
    for (char* src = s; *src; ++src) {
        if (isalpha((unsigned char)*src) || isspace((unsigned char)*src))
            *dst++ = tolower((unsigned char)*src);
    }
    *dst = '\0';
}

// Thread-safe tokenize using strtok_r; returns word count
int tokenize(char* line, char* words[], int maxw) {
    int n = 0;
    char* saveptr;
    char* tok = strtok_r(line, " \t\r\n", &saveptr);
    while (tok && n < maxw) {
        words[n++] = tok;
        tok = strtok_r(NULL, " \t\r\n", &saveptr);
    }
    return n;
}

// Return 1 if phrase (one or more words) exists in words[]
int phrase_match(char* words[], int wc, const char* phrase) {
    char buf[MAX_LINE_LENGTH];
    strncpy(buf, phrase, sizeof buf);
    buf[sizeof buf -1] = '\0';
    normalize(buf);

    char* pw[MAX_WORDS_PER_LINE];
    int pwc = tokenize(buf, pw, MAX_WORDS_PER_LINE);
    if (pwc == 0) return 0;

    for (int i = 0; i <= wc - pwc; ++i) {
        int ok = 1;
        for (int j = 0; j < pwc; ++j) {
            if (strcmp(words[i + j], pw[j]) != 0) { ok = 0; break; }
        }
        if (ok) return 1;
    }
    return 0;
}

// ——— OpenMP Search + Logging —————————————————————————————————————

void search_and_log_openmp(const char* filename, char* phrases[], int pc) {
    // 1) read all lines
    FILE* f = fopen(filename, "r");
    if (!f) { perror("Error opening file"); return; }
    int cap = 1024, line_count = 0;
    char** lines = malloc(cap * sizeof *lines);
    char buf[MAX_LINE_LENGTH];
    while (fgets(buf, sizeof buf, f)) {
        if (line_count >= cap) {
            cap *= 2;
            lines = realloc(lines, cap * sizeof *lines);
            if (!lines) { fprintf(stderr,"OOM\n"); exit(1); }
        }
        lines[line_count++] = strdup(buf);
    }
    fclose(f);

    int counts[MAX_PHRASES] = {0}, total = 0;

    // 2) parallel region
    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        int lc[MAX_PHRASES] = {0}, lt = 0;
        #pragma omp for schedule(static)
        for (int i = 0; i < line_count; ++i) {
            // copy & normalize
            char line_copy[MAX_LINE_LENGTH];
            strncpy(line_copy, lines[i], sizeof line_copy);
            line_copy[sizeof line_copy -1] = '\0';
            normalize(line_copy);

            // tokenize
            char* words[MAX_WORDS_PER_LINE];
            int wc = tokenize(line_copy, words, MAX_WORDS_PER_LINE);

            // match
            for (int j = 0; j < pc; ++j) {
                if (phrase_match(words, wc, phrases[j])) {
                    lc[j]++; lt++;
                }
            }
        }
        // reduce
        #pragma omp critical
        {
            for (int j = 0; j < pc; ++j) counts[j] += lc[j];
            total += lt;
        }
    }
    double t1 = omp_get_wtime();

    // 3) ASCII table
    printf("\n+-------------------------------+------------+\n");
    printf("| %-29s | %10s |\n", "Phrase", "Matches");
    printf("+-------------------------------+------------+\n");
    for (int i = 0; i < pc; ++i) {
        printf("| %-29s | %10d |\n", phrases[i], counts[i]);
    }
    printf("+-------------------------------+------------+\n");
    printf("| %-29s | %10d |\n", "Total matches", total);
    printf("+-------------------------------+------------+\n");
    printf("| %-29s | %10.4f |\n", "Elapsed time (s)", t1 - t0);
    printf("+-------------------------------+------------+\n");

    // 4) CSV log
    {
        static int header = 0;
        FILE* csv = fopen("results.csv","a");
        if (csv) {
            if (!header) {
                fprintf(csv,
                    "timestamp,filename,phrases,total_matches,time_s\n");
                header = 1;
            }
            // build list
            char plist[MAX_INPUT_SIZE] = "";
            for (int i = 0; i < pc; ++i) {
                if (i) strcat(plist,";");
                strcat(plist, phrases[i]);
            }
            char ts[64];
            time_t now=time(NULL);
            strftime(ts,sizeof ts,"%Y-%m-%d %H:%M:%S",localtime(&now));
            fprintf(csv,
                "\"%s\",\"%s\",\"%s\",%d,%.4f\n",
                ts, filename, plist, total, t1 - t0);
            fclose(csv);
        }
    }

    // 5) cleanup
    for (int i = 0; i < line_count; ++i) free(lines[i]);
    free(lines);
}

// ——— Main / UI —————————————————————————————————————————

int main(void) {
    char filepath[MAX_INPUT_SIZE],
         phrase_line[MAX_INPUT_SIZE],
         *phrases[MAX_PHRASES];
    int pc = 0;

    printf("Enter path to text file: ");
    if (!fgets(filepath,sizeof filepath,stdin)) return 0;
    filepath[strcspn(filepath,"\r\n")] = 0;

    printf("Enter search phrases, comma-separated:\n");
    if (!fgets(phrase_line,sizeof phrase_line,stdin)) return 0;
    phrase_line[strcspn(phrase_line,"\r\n")] = 0;

    // proper strtok_r usage
    char *saveptr, *tok = strtok_r(phrase_line, ",", &saveptr);
    while (tok && pc < MAX_PHRASES) {
        // trim
        while (*tok==' ') ++tok;
        char* end = tok+strlen(tok)-1;
        while (end>tok && *end==' ') *end--='\0';
        if (*tok) phrases[pc++] = tok;
        tok = strtok_r(NULL, ",", &saveptr);
    }
    if (pc==0) { printf("No valid phrases entered. Exiting..\n"); return 0; }

    printf("Loaded %d threads. Starting search...\n",
            omp_get_max_threads());
    search_and_log_openmp(filepath, phrases, pc);
    return 0;
}
