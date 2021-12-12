//! U-D分解フィルタの実装（最適化実装版）
//!
//! 参考：片山 徹，”応用カルマンフィルタ”，朝倉書店，pp. 152-154，1983．
//! 
//! このプログラムでは，観測ノイズの共分散行列Rを対角行列のみに限定している．
//! 多くの場合は各観測値が独立で対角行列となるし，仮に対角でない場合も
//! V = R*R^Tを求め，y:=inverse(V)*y, H:=inverse(V)*H, R:=Iとおけば良い．

use std::mem::MaybeUninit;

// --------- 入出力数 --------- //
/// 状態変数の個数
const SYS_N: usize = 3;

/// 入力数
const SYS_R: usize = 2;

/// 出力数
const SYS_P: usize = 2;
// ---------------------------- //

// -- ベクトル・行列型の定義 -- //
// Vector○: X次元ベクトル
type VectorN<T>  = [T; SYS_N];
type VectorP<T>  = [T; SYS_P];
type VectorNR<T> = [T; SYS_N + SYS_R];
// Matrix○x□: ○行□列行列
type MatrixNxN<T>  = [[T; SYS_N]; SYS_N];
type MatrixNxR<T>  = [[T; SYS_R]; SYS_N];
type MatrixPxN<T>  = [[T; SYS_N]; SYS_P];
type MatrixRxR<T>  = [[T; SYS_R]; SYS_R];
type MatrixNxNR<T> = [[T; SYS_N + SYS_R]; SYS_N];  // N x (N+R)
// ---------------------------- //

fn main() {
    // --- 各行列を定義 --- //
    let p = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ];
    let f = [
        [0.0, 0.0, -2.72],
        [1.0, 0.0,  2.466],
        [0.0, 1.0, -0.7452]
    ];
    let g = [
        [0.0, 1.0],
        [0.0, 0.0],
        [1.0, 0.0]
    ];
    let h = [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ];
    let q = [
        [1.0, 0.0],
        [0.0, 1.0]
    ];
    let r = [1.0, 1.0];
    let mut ud_filter = UdFilter::new(p, f, g, h, q, r);

    /* 20ループで以下の値に収束する
    UD: Matrix3x3 = [
        [2.863648175007036, -0.564064186468,     0.654853726808],
        [0.000000000000,     6.195463220228957, -0.857728817252],
        [0.000000000000,     0.000000000000,     2.3501025708970573],
    ];
    */
    println!("Loop start!");
    for _ in 0..20 {
        
        ud_filter.predict();
        plot_nn(&ud_filter.U);
        ud_filter.filtering(&[0.0, 0.0]);
        
    }
    
}

/// U-D分解フィルタ
/// 
/// 誤差共分散行列の初期値は零行列以外にしないとU-D分解に失敗するので，
/// スカラー行列にするのが無難．
#[allow(non_snake_case)]
struct UdFilter {
    pub x: VectorN<f64>,    // 状態変数
    pub U: MatrixNxN<f64>,  // U-D分解した共分散行列
    F: MatrixNxN<f64>,  // システム行列
    G: MatrixNxR<f64>,  // 入力行列
    H: MatrixPxN<f64>,  // 出力行列
    R: VectorP<f64>,    // 観測ノイズの共分散行列の対角成分
}

impl UdFilter {
    #[allow(non_snake_case)]
    pub fn new(
        P: MatrixNxN<f64>,  // 共分散行列の初期値
        F: MatrixNxN<f64>,  // システム行列
        G: MatrixNxR<f64>,  // 入力行列
        H: MatrixPxN<f64>,  // 出力行列
        Q: MatrixRxR<f64>,  // システムノイズの共分散行列
        R: VectorP<f64>     // 観測ノイズの共分散行列の対角成分
    ) -> Self {
        // システムノイズの分散Qが単位行列で無い場合には，
        // QをQ=C*C^Tと分解し，Gを改めてG*Cとおく．
        let c = cholesky_decomp(Q);
        let mut gc: MatrixNxR<f64> = unsafe {MaybeUninit::uninit().assume_init()};
        for i in 0..SYS_N {
            for j in 0..SYS_R {
                let mut sum = 0.0;
                for k in 0..SYS_R {
                    sum += G[i][k] * c[k][j];
                }
                gc[i][j] = sum;
            }
        }
        Self {
            x: [0.0; SYS_N],
            U: ud_decomp(P),
            F: F,
            G: gc,
            H: H,
            R: R,
        }
    }

    /// 予測ステップ
    /// 
    /// x(k+1) = F * x(k)
    /// P_bar = F*P*F^T + G*Q*Q^T
    pub fn predict(&mut self) {
        // Working array
        let mut qq: VectorN<f64>    = unsafe {MaybeUninit::uninit().assume_init()};
        let mut v:  VectorNR<f64>   = unsafe {MaybeUninit::uninit().assume_init()};
        let mut z:  VectorNR<f64>   = unsafe {MaybeUninit::uninit().assume_init()};
        let mut w:  MatrixNxNR<f64> = unsafe {MaybeUninit::uninit().assume_init()};

        for i in 0..SYS_N {
            let mut sum = 0.0;
            for j in 0..SYS_N {
                sum += self.F[i][j] * self.x[j];
            }
            v[i] = sum;
        }
        for j in (1..SYS_N).rev() {
            qq[j] = self.U[j][j];
            self.x[j] = v[j];
            for i in 0..SYS_N {
                let mut sum = self.F[i][j];
                for k in 0..j {
                    sum += self.F[i][k] * self.U[k][j];
                }
                w[i][j] = sum;
            }
        }
        for i in 0..SYS_N {
            for j in 0..SYS_R {
                w[i][j + SYS_N] = self.G[i][j];
            }
            w[i][0] = self.F[i][0];
        }
        qq[0] = self.U[0][0];
        self.x[0] = v[0];
        // --- ここまででw, qq, self.xを計算

        for j in (1..SYS_N).rev() {
            let mut sum = 0.0;
            for k in 0..SYS_N {
                v[k] = w[j][k];
                z[k] = v[k] * qq[k];
                sum += z[k] * v[k];
            }
            for k in SYS_N..(SYS_N + SYS_R) {
                v[k] = w[j][k];
                z[k] = v[k];
                sum += v[k] * v[k];
            }
            self.U[j][j] = sum;
            let u_recip = self.U[j][j].recip();
            for i in 0..j {
                sum = 0.0;
                for k in 0..(SYS_N + SYS_R) {
                    sum += w[i][k] * z[k];
                }
                sum *= u_recip;
                for k in 0..(SYS_N + SYS_R) {
                    w[i][k] -= sum * v[k];
                }
                self.U[i][j] = sum;
            }
        }
        let mut sum = 0.0;
        for k in 0..SYS_N {
            sum += qq[k] * (w[0][k] * w[0][k]);
        }
        for k in SYS_N..(SYS_N + SYS_R) {
            sum += w[0][k] * w[0][k];
        }
        self.U[0][0] = sum;
    }

    /// フィルタリングステップ
    pub fn filtering(&mut self, y: &VectorP<f64>) {
        // Working array
        let mut ff: VectorN<f64> = unsafe {MaybeUninit::uninit().assume_init()};  // U^T H^T
        let mut gg: VectorN<f64> = unsafe {MaybeUninit::uninit().assume_init()};  // D U^T H^T

        // 出力の数だけループ
        for l in 0..SYS_P {
            let mut y_diff = y[l];  // y_diff := y - H*x
            for j in 0..SYS_N {
                y_diff -=  self.H[l][j] * self.x[j];
            }
            for j in (1..SYS_N).rev() {
                ff[j] = self.H[l][j];
                for k in 0..j {
                    ff[j] += self.U[k][j] * self.H[l][k];
                }
                gg[j] = self.U[j][j] * ff[j];
            }
            ff[0] = self.H[l][0];
            gg[0] = self.U[0][0] * ff[0];
            // --- ここまででy_diff, ff, ggを計算

            let mut alpha = self.R[l] + ff[0] * gg[0];  // 式 8.46
            let mut gamma = alpha.recip();
            self.U[0][0] = self.R[l] * gamma * self.U[0][0];  // 式 8.46
            for j in 1..SYS_N {
                let mut beta = alpha;
                alpha += ff[j] * gg[j];  // 式 8.47
                let lambda = ff[j] * gamma;  // 式　8.49
                gamma = alpha.recip();
                self.U[j][j] = beta * self.U[j][j] * gamma;  // 式 8.48
                for i in 0..j {
                    beta = self.U[i][j];
                    self.U[i][j] -= lambda * gg[i];  // 式 8.50
                    gg[i] +=  beta * gg[j];  // 式 8.51
                }
            }
            y_diff *= gamma;
            for j in 0..SYS_N {
                self.x[j] += gg[j] * y_diff;
            }
        }
    }
}


/// U-D分解（P = U * D * U^T）
/// 
/// * Pをn×n非負正定値対称行列とする．
/// * Uは対角成分を1とするn×n上三角行列．
/// * Dはn×n対角行列．
/// 
/// 返り値は，対角成分をDとし，それ以外の要素をUとした上三角行列．
fn ud_decomp(mut p: MatrixNxN<f64>) -> MatrixNxN<f64> {
    let mut ud: MatrixNxN<f64> = unsafe {MaybeUninit::uninit().assume_init()};

    for k in (1..SYS_N).rev() {  // n-1, n-2, ..., 1
        ud[k][k] = p[k][k];
        let ud_recip = ud[k][k].recip();
        for j in 0..k {
            ud[j][k] = p[j][k] * ud_recip;
            ud[k][j] = 0.0;  // 対角を除いた下三角成分を0埋め

            let tmp = ud[j][k] * ud[k][k];
            for i in 0..=j {  // 両側閉区間
                p[i][j] -= ud[i][k] * tmp;
            }
        }
    }
    ud[0][0] = p[0][0];  // pを書き換えてるから，d[0]の代入はこの位置じゃないとダメ

    ud
}

/// コレスキー分解（P = U * U^T）
/// 
/// * Pをn×n非負正定値対称行列とする．
/// * Uは対角要素が非負の値をとるn×n上三角行列．
fn cholesky_decomp(mut p: MatrixRxR<f64>) -> MatrixRxR<f64> {
    let mut u: MatrixRxR<f64> = unsafe {MaybeUninit::uninit().assume_init()};

    for k in (1..SYS_R).rev() {
        u[k][k] = p[k][k].sqrt();
        let u_recip = u[k][k].recip();
        for j in 0..k {
            u[j][k] = p[j][k] * u_recip;
            u[k][j] = 0.0;  // 対角を除いた下三角成分を0埋め
            for i in 0..=j {
                p[i][j] -= u[i][k] * u[j][k];
            }
        }
    }
    u[0][0] = p[0][0].sqrt();

    u
}

/// MatrixNxNを整形してプロット
fn plot_nn(m: &MatrixNxN<f64>) {
    println!("MatrixNxN = [");
    for i in 0..SYS_N {
        print!("    [");
        for j in 0..SYS_N {
            print!("{:.12}", m[i][j]);
            if j < (SYS_N - 1) {
                print!(", ");
            }
        }
        println!("],");
    }
    println!("];");
}

#[test]
fn test_ud() {
    let p = [
        [10.0,  7.0, 2.0],
        [7.0,  10.0, 2.0],
        [2.0,   2.0, 4.0]
    ];

    let ud = ud_decomp(p);
    let ud_true = [
        [5.0, 2.0/3.0, 1.0/2.0],
        [0.0,     9.0, 1.0/2.0],
        [0.0,     0.0,     4.0],
    ];
    for i in 0..SYS_N {
        for j in 0..SYS_N {
            assert!((ud[i][j] - ud_true[i][j]).abs() < 1e-7);
        }
    }
}

#[test]
fn test_cholesky() {
    let p = [
        [10.0,  7.0],
        [7.0,  10.0],
    ];

    let c = cholesky_decomp(p);
    let c_true = [
        [((100.0 - 49.0)/10.0f64).sqrt(), 7.0/10.0f64.sqrt()],
        [0.0, 10.0f64.sqrt()]
    ];
    for i in 0..SYS_R {
        for j in 0..SYS_R {
            assert!((c[i][j] - c_true[i][j]).abs() < 1e-5);
        }
    }
}