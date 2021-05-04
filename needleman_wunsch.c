#include <stdio.h>
#include <stdlib.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

#define MAX(a,b) (((a)>(b))?(a):(b))
#define FIT_SCORE(a, b) (a == b ? MATCH : MISMATCH)

/*
 * MAX OF (FIT (MATCH OR MISMATCH), INSERT, DELETE) SCORES
 */
int compute_score_at_index(int **M, char a, char b, int i, int j) {
    int score = M[i-1][j-1] + FIT_SCORE(a, b);
    int insert = M[i][j-1] + GAP;
    int delete = M[i-1][j] + GAP;

    int max = MAX(score, insert);
    max = MAX(max, delete);
 
    return max;
}

int **needleman_wunsch_score(char *seqA, char *seqB, int lenA, int lenB) {
    int i, j, **score;
    char a, b;
    
    // alocare memorie
    score = (int **) calloc((lenA + 1), sizeof(int *));
    for (i = 0; i <= lenA; i++) {
        score[i] = (int *) calloc((lenB + 1), sizeof(int));
    }

    // initializat prima coloana
    // all gaps => indexului coloanei * GAP score
    for (i = 0; i <= lenA; i++) {
        score[i][0] = i * GAP;
    }

    // initializat prima linie
    // all gaps => indexului liniei * GAP score
    for (j = 0; j <= lenB; j++) {
        score[0][j] = j * GAP;
    }

    int m = lenA + 1;
    int n = lenB + 1;

    // traversare anti-diagonala, exceptand prima linie si prima coloana
    for (int slice = 2; slice < m + n - 1; slice++) {
        int z1 = slice <= n ? 1 : slice - n + 1;
        int z2 = slice <= m ? 1 : slice - m + 1;
        for (j = slice - z2; j >= z1; j--) {
            a = seqA[j-1];
            b = seqB[slice-j-1];
            score[j][slice-j] = compute_score_at_index(score, a, b, j, slice-j);
        }
    }

    return score;
}

void needleman_wunsch_align(char *seqA, char *seqB, int lenA, int lenB) {
    int **score = needleman_wunsch_score(seqA, seqB, lenA, lenB);
    int i, j, k;
    char *alignedA, *alignedB;

    // afiseaza matricea de scor 
    for (i = 0; i <= lenA; i++) {
        for (j = 0; j <= lenB; j++) {
            printf("%3d ", score[i][j]);
        }
        printf("\n");
    }

    // alocare memorie
    k = lenA + lenB; // alinierea poate sa fie de lungime maxima lenA + lenB
    alignedA = (char *) calloc((k + 1), sizeof(char *));
    alignedA[k] = '\0';
    alignedB = (char *) calloc((k + 1), sizeof(char *));
    alignedB[k] = '\0';
    k--;

    i=lenA;
    j=lenB;

    while (i > 0 && j > 0) {
        if (score[i][j] == score[i-1][j-1] + FIT_SCORE(seqA[i-1], seqB[j-1])) {
            alignedA[k] = seqA[i-1];
            alignedB[k] = seqB[j-1];
            i--;
            j--;
        }
        else if (score[i][j] == score[i-1][j] + GAP) {
            alignedA[k] = seqA[i-1];
            alignedB[k] = '-';
            i--;
        }
        else {
            alignedA[k] = '-';
            alignedB[k] = seqB[j-1];
            j--;

        }
        k--;
    }

    while (i > 0 || j > 0) {
        if (i == 0) {
            alignedA[k] = '-';
            alignedB[k] = seqB[j-1];
            j--;
        }
        else if (j == 0) {
            alignedA[k] = seqA[i-1];
            alignedB[k] = '-';
            i--;
        }
        k--;
    }
    
    // afisare secvente aliniate
    printf("\nInitial:\n%s\n%s\n\n", alignedA+k+1, alignedB+k+1);
    printf("Aliniere:\n%s\n%s\n\n", alignedA+k+1, alignedB+k+1);

    // dealocare memorie
    for (i = 0; i <= lenA; i++) {
        free(score[i]);
    }
    free(score);

    free(alignedA);
    free(alignedB);
}

int main() {
    // TODO Read from file
    needleman_wunsch_align("CAGCTAGCG", "CCATACGA", 9, 8);
    return 0;
}