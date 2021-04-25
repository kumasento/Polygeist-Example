void MatrixMult(float A[100][200], float B[200][300], float C[100][300]) {
  int i, j, k;

  for (i = 0; i < 100; i ++) {
    for (j = 0; j < 300; j ++) {
      C[i][j] = 0;
      for (k = 0; k < 200; k ++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
