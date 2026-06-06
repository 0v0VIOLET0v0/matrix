#include<iostream>
#include<vector>
#include<algorithm>
#include<chrono>
#include<numeric>
#include<limits>

constexpr int M = 256;
constexpr int N = 256;
constexpr int K = 256;

constexpr int BM = 8;
constexpr int BN = 8;
constexpr int BK = 16;

void mul(const float *a, const float *b, float *c,int lda,int ldb,int ldc) //8*16*8矩阵乘法
{
    for(int i=0;i<BM;i++)
    {
        for(int j=0;j<BN;j++)
        {
            float sum=0.0f;
            for(int k=0;k<BK;k++)
            {
                sum+=a[i*lda+k]*b[k*ldb+j];
            }
            c[i*ldc+j]+=sum;
        }
    }
    return;
}

void gemm_tail(
    const float* a,
    const float* b,
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
                sum += a[i * lda + k] * b[k * ldb + j];
            }

            c[i * ldc + j] += sum;
        }
    }
}

void gemm_256(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c,int m,int n,int k) //256*256*256矩阵乘法
{
    std::fill(c.begin(), c.end(), 0.0f);
    int m_full = m / BM * BM;
    int n_full = n / BN * BN;
    int k_full = k / BK * BK;
    for(int i=0;i<m_full;i+=BM)
    {
        for(int j=0;j<n_full;j+=BN)
        {
            for(int k=0;k<k_full;k+=BK)
            {
                mul(&a[i*k+k],&b[k*n+j],&c[i*n+j],k,n,n);
            }
            if (k_full < k) {
                gemm_tail(
                    &a[i * k + k_full],
                    &b[k_full * n + j],
                    &c[i * n + j],
                    BM,
                    BN,
                    k - k_full,
                    k,
                    n,
                    n
                );
            }
        }
        
    }
    // 右侧尾部：完整行块 + 剩余列
    if (n_full < n) {
        for (int i = 0; i < m_full; i += BM) {
            gemm_tail(
                &a[i * k],
                &b[n_full],
                &c[i * n + n_full],
                BM,
                n - n_full,
                k,
                k,
                n,
                n
            );
        }
    }

    // 下方尾部：剩余行 + 完整列块
    if (m_full < m) {
        for (int j = 0; j < n_full; j += BN) {
            gemm_tail(
                &a[m_full * k],
                &b[j],
                &c[m_full * n + j],
                m - m_full,
                BN,
                k,
                k,
                n,
                n
            );
        }
    }

    // 右下角尾部：剩余行 + 剩余列
    if (m_full < m && n_full < n) {
        gemm_tail(
            &a[m_full * k],
            &b[n_full],
            &c[m_full * n + n_full],
            m - m_full,
            n - n_full,
            k,
            k,
            n,
            n
        );
    }
}

double calc_gops(int m,int n,int k,double seconds)
{
    return 2.0*m*n*k/seconds/1e9;
}

void benchmark_gemm_256(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) //计算参数
{
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 256;

    int warmup = 3;
    int repeat = 20;

    for (int r = 0; r < warmup; r++) {
        gemm_256(A, B, C, M, N, K);
    }

    double total_time = 0.0;
    double best_time = std::numeric_limits<double>::max();

    for (int r = 0; r < repeat; r++) {
        auto start = std::chrono::steady_clock::now();

        gemm_256(A, B, C, M, N, K);

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


int main()
{
    int m = 247;
    int n = 231;
    int k = 256;

    std::vector<float> A(m * k, 1.0f);
    std::vector<float> B(k * n, 1.0f);
    std::vector<float> C(m * n, 0.0f);

    gemm_256(A, B, C, m, n, k);

    std::cout << "C[0][0] = " << C[0] << "\n";
    std::cout << "C[last] = " << C[(m - 1) * n + (n - 1)] << "\n";

    return 0;
}
