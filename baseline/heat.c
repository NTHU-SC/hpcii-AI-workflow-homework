/*
 * heat.c -- 2D heat diffusion on a square plate (temperatures in kelvin).
 *
 * The plate is an N x N grid. The outermost ring of cells is the boundary:
 * each side is held at a fixed temperature for the whole run. Interior
 * cells start at one initial temperature, and rectangular heat sources add
 * a fixed amount of heat to the cells they cover on every step.
 *
 * Each step applies the explicit 5-point diffusion update to every interior
 * cell, reading only the previous step's field (Jacobi style):
 *
 *     T'[c] = T[c] + A * (T[west] + T[east] + T[south] + T[north] - 4 T[c]) + Q[c]
 *
 * with A = 0.2 (stable for A <= 0.25). After STEPS steps the final field is
 * written to OUTPUT as N*N little-endian float64 values, row by row
 * (row 0 is the bottom edge, column 0 the left edge).
 *
 * Usage:  heat INPUT OUTPUT [-n N] [-s STEPS]
 *
 * Input file (lines starting with '#' are comments):
 *     left right bottom top      boundary temperatures (K)
 *     initial                    initial interior temperature (K)
 *     nsrc                       number of heat sources
 *     x0 y0 x1 y1 q              one line per source: a rectangle given as
 *                                fractions of the plate in [0,1], and the
 *                                heat added per step (K) to each covered cell
 *
 * Standard output:
 *     config: n=<N> steps=<STEPS> threads=<OpenMP threads>
 *     time: <seconds> s          wall-clock time of the step loop only
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define A 0.2

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Read the next number, skipping whitespace and '#' comment lines. */
static int next_number(FILE *f, double *v)
{
    int ch;
    for (;;) {
        ch = fgetc(f);
        if (ch == EOF) return 0;
        if (ch == '#') {
            while (ch != '\n' && ch != EOF) ch = fgetc(f);
            continue;
        }
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') continue;
        ungetc(ch, f);
        return fscanf(f, "%lf", v) == 1;
    }
}

static void die(const char *msg)
{
    fprintf(stderr, "heat: %s\n", msg);
    exit(1);
}

int main(int argc, char **argv)
{
    int n = 512, steps = 8000;
    const char *in_path = NULL, *out_path = NULL;

    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-n") && a + 1 < argc) n = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-s") && a + 1 < argc) steps = atoi(argv[++a]);
        else if (!in_path) in_path = argv[a];
        else if (!out_path) out_path = argv[a];
        else die("usage: heat INPUT OUTPUT [-n N] [-s STEPS]");
    }
    if (!in_path || !out_path) die("usage: heat INPUT OUTPUT [-n N] [-s STEPS]");
    if (n < 3 || steps < 1) die("need n >= 3 and steps >= 1");

    /* ---- read the input ---- */
    FILE *f = fopen(in_path, "r");
    if (!f) die("cannot open input file");
    double left, right, bottom, top, initial, nsrc_d;
    if (!next_number(f, &left) || !next_number(f, &right) ||
        !next_number(f, &bottom) || !next_number(f, &top) ||
        !next_number(f, &initial) || !next_number(f, &nsrc_d))
        die("malformed input file");

    size_t cells = (size_t)n * n;
    double *T  = malloc(cells * sizeof *T);
    double *Tn = malloc(cells * sizeof *Tn);
    double *Q  = calloc(cells, sizeof *Q);
    if (!T || !Tn || !Q) die("out of memory");

    int nsrc = (int)nsrc_d;
    for (int s = 0; s < nsrc; s++) {
        double x0, y0, x1, y1, q;
        if (!next_number(f, &x0) || !next_number(f, &y0) || !next_number(f, &x1) ||
            !next_number(f, &y1) || !next_number(f, &q))
            die("malformed heat source line");
        int i0 = (int)(x0 * (n - 1)), i1 = (int)(x1 * (n - 1));
        int j0 = (int)(y0 * (n - 1)), j1 = (int)(y1 * (n - 1));
        if (i0 < 1) i0 = 1;
        if (j0 < 1) j0 = 1;
        if (i1 > n - 2) i1 = n - 2;
        if (j1 > n - 2) j1 = n - 2;
        for (int j = j0; j <= j1; j++)
            for (int i = i0; i <= i1; i++)
                Q[(size_t)j * n + i] += q;
    }
    fclose(f);

    /* ---- initial field: interior at `initial`, sides at their fixed values ---- */
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            double v = initial;
            if (i == 0) v = left;
            if (i == n - 1) v = right;
            if (j == 0) v = bottom;     /* bottom and top rows own the corners */
            if (j == n - 1) v = top;
            T[(size_t)j * n + i] = v;
        }
    memcpy(Tn, T, cells * sizeof *T);

    printf("config: n=%d steps=%d threads=%d\n", n, steps, omp_get_max_threads());
    fflush(stdout);

    /* ---- timed region: the step loop ---- */
    double t0 = now();
    for (int step = 0; step < steps; step++) {
        #pragma omp parallel for schedule(static)
        for (int j = 1; j < n - 1; j++) {
            const size_t row = (size_t)j * n;
            for (int i = 1; i < n - 1; i++) {
                const size_t c = row + i;
                double sum = T[c - 1] + T[c + 1] + T[c - n] + T[c + n];
                Tn[c] = T[c] + A * (sum - 4.0 * T[c]) + Q[c];
            }
        }
        double *tmp = T; T = Tn; Tn = tmp;
    }
    double t1 = now();

    printf("time: %.3f s\n", t1 - t0);
    fflush(stdout);

    /* ---- write the final field ---- */
    FILE *o = fopen(out_path, "wb");
    if (!o) die("cannot open output file");
    if (fwrite(T, sizeof *T, cells, o) != cells) die("short write");
    if (fclose(o) != 0) die("cannot close output file");

    free(T); free(Tn); free(Q);
    return 0;
}
