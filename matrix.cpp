#include<iostream>
#include<vector>
#include<algorithm>
#include<chrono>
#include<numeric>
#include<limits>
#include<cmath>
#include<random>

constexpr int M = 256;
constexpr int N = 256;
constexpr int K = 256;

constexpr int BM = 8;
constexpr int BN = 8;
constexpr int BK = 16;

void mul(const float *a, const float *bt, float *c,int lda,int ldb,int ldc) //8*16*8 matrix multiply with transposed B
{
    for(int i=0;i<BM;i++)
    {
        const float* arow = a + i * lda;
        if (i + 1 < BM) {
            __builtin_prefetch(a + (i + 1) * lda, 0, 1);
        }

        for(int j=0;j<BN;j++)
        {
            const float* brow = bt + j * ldb;
            if (j + 1 < BN) {
                __builtin_prefetch(bt + (j + 1) * ldb, 0, 1);
            }

            float sum = 0.0f;
            sum += arow[0] * brow[0];
            sum += arow[1] * brow[1];
            sum += arow[2] * brow[2];
            sum += arow[3] * brow[3];
            sum += arow[4] * brow[4];
            sum += arow[5] * brow[5];
            sum += arow[6] * brow[6];
            sum += arow[7] * brow[7];
            sum += arow[8] * brow[8];
            sum += arow[9] * brow[9];
            sum += arow[10] * brow[10];
            sum += arow[11] * brow[11];
            sum += arow[12] * brow[12];
            sum += arow[13] * brow[13];
            sum += arow[14] * brow[14];
            sum += arow[15] * brow[15];
            c[i*ldc+j] += sum;
        }
    }
    return;
}

void gemm_tail(
    const float* a,
    const float* bt,
    float* c,
    int rows,
    int cols,
    int depth,
    int lda,
    int ldb,
    int ldc
) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float sum = 0.0f;

            for (int k = 0; k < depth; k++) {
                sum += a[i * lda + k] * bt[j * ldb + k];
            }

            c[i * ldc + j] += sum;
        }
    }
}

void gemm_256(const std::vector<float> &a, const std::vector<float> &bt, std::vector<float> &c,int m,int n,int k) //256*256*256 matrix multiply with transposed B
{
    std::fill(c.begin(), c.end(), 0.0f);
    int m_full = m / BM * BM;
    int n_full = n / BN * BN;
    int k_full = k / BK * BK;
    for(int i=0;i<m_full;i+=BM)
    {
        for(int j=0;j<n_full;j+=BN)
        {
            for(int p=0;p<k_full;p+=BK)
            {
                mul(&a[i*k+p],&bt[j*k+p],&c[i*n+j],k,k,n);
            }
            if (k_full < k) {
                gemm_tail(
                    &a[i * k + k_full],
                    &bt[j * k + k_full],
                    &c[i * n + j],
                    BM,
                    BN,
                    k - k_full,
                    k,
                    k,
                    n
                );
            }
        }

    }
    if (n_full < n) {
        for (int i = 0; i < m_full; i += BM) {
            gemm_tail(
                &a[i * k],
                &bt[n_full * k],
                &c[i * n + n_full],
                BM,
                n - n_full,
                k,
                k,
                k,
                n
            );
        }
    }

    if (m_full < m) {
        for (int j = 0; j < n_full; j += BN) {
            gemm_tail(
                &a[m_full * k],
                &bt[j * k],
                &c[m_full * n + j],
                m - m_full,
                BN,
                k,
                k,
                k,
                n
            );
        }
    }

    if (m_full < m && n_full < n) {
        gemm_tail(
            &a[m_full * k],
            &bt[n_full * k],
            &c[m_full * n + n_full],
            m - m_full,
            n - n_full,
            k,
            k,
            k,
            n
        );
    }
}

double calc_gops(int m,int n,int k,double seconds)
{
    return 2.0*m*n*k/seconds/1e9;
}

void benchmark_gemm_256(const std::vector<float>& A, const std::vector<float>& B_t, std::vector<float>& C)
{
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 256;

    int warmup = 3;
    int repeat = 20;

    for (int r = 0; r < warmup; r++) {
        gemm_256(A, B_t, C, M, N, K);
    }

    double total_time = 0.0;
    double best_time = std::numeric_limits<double>::max();

    for (int r = 0; r < repeat; r++) {
        auto start = std::chrono::steady_clock::now();

        gemm_256(A, B_t, C, M, N, K);

        auto end = std::chrono::steady_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count();

        total_time += seconds;
        best_time = std::min(best_time, seconds);
    }

    double avg_time = total_time / repeat;

    double avg_gops = calc_gops(M, N, K, avg_time);
    double best_gops = calc_gops(M, N, K, best_time);

    float checksum = std::accumulate(C.begin(), C.end(), 0.0f);

    std::cout << "avg time  = " << avg_time << " s\n";
    std::cout << "best time = " << best_time << " s\n";
    std::cout << "avg GOPS  = " << avg_gops << "\n";
    std::cout << "best GOPS = " << best_gops << "\n";
    std::cout << "checksum  = " << checksum << "\n";
}

void matmul_ref(
    const std::vector<float>& a,
    const std::vector<float>& bt,
    std::vector<float>& c,
    int m,
    int n,
    int k
) {
    std::fill(c.begin(), c.end(), 0.0f);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;

            for (int p = 0; p < k; p++) {
                sum += a[i * k + p] * bt[j * k + p];
            }

            c[i * n + j] = sum;
        }
    }
}

void transpose_b(
    const std::vector<float>& b,
    std::vector<float>& bt,
    int n,
    int k
) {
    for (int p = 0; p < k; p++) {
        for (int j = 0; j < n; j++) {
            bt[j * k + p] = b[p * n + j];
        }
    }
}

void fill_random(std::vector<float>& x, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    for (float& v : x) {
        v = dist(gen);
    }
}

bool check_result(
    const std::vector<float>& c,
    const std::vector<float>& ref,
    float eps = 1e-3f
) {
    for (size_t i = 0; i < c.size(); i++) {
        float diff = std::fabs(c[i] - ref[i]);

        if (diff > eps) {
            std::cout << "Mismatch at " << i
                      << ": got " << c[i]
                      << ", ref " << ref[i]
                      << ", diff " << diff << "\n";
            return false;
        }
    }

    return true;
}

int main()
{
    int m = 247;
    int n = 231;
    int k = 256;

    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> B_t(n * k);
    std::vector<float> C(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);

    std::mt19937 gen(0);
    fill_random(A, gen);
    fill_random(B, gen);
    transpose_b(B, B_t, n, k);

    gemm_256(A, B_t, C, m, n, k);
    matmul_ref(A, B_t, C_ref, m, n, k);

    if (check_result(C, C_ref)) {
        std::cout << "Correct\n";
    } else {
        std::cout << "Wrong\n";
    }

    return 0;
}
