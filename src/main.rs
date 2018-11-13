extern crate time;
extern crate rayon;
extern crate num_cpus;

use rayon::prelude::*;
use rayon::ThreadPool;
use rayon::ThreadPoolBuildError;

const LOOP: usize = 10;
const P: i64 = 6_700_417;
const MU: i64 = 22;
const NUM_VALUES: usize = 0x2_000_000;

fn logisticsmap_calc(x: i64, p: i64, mu: i64) -> i64 {
    mu * x * (x + 1) % p
}

fn logisticsmap_loop_calc(num: usize, x: i64, p: i64, mu: i64) -> i64 {
    (0..num).fold(x, |x, _| logisticsmap_calc(x, p, mu))
}

fn set_num_threads(n: usize) -> Result<ThreadPool, ThreadPoolBuildError> {
  rayon::ThreadPoolBuilder::new().num_threads(n).build()
}

fn benchmark_cpu_multi() -> i64 {
    let r1 = (0..NUM_VALUES as i64).collect::<Vec<_>>().par_iter().map(|&x| logisticsmap_loop_calc(LOOP, x, P, MU)).collect::<Vec<i64>>();
    r1[10000]
}

fn benchmark(n: usize) -> Result<f64, ThreadPoolBuildError> {
    {
    		match set_num_threads(n) {
    			Ok(pool) => pool.install(|| {
        		let start_time = time::get_time();
        		benchmark_cpu_multi();
        		let end_time = time::get_time();
        		let diffsec = end_time.sec - start_time.sec;   // i64
        		let diffsub = end_time.nsec - start_time.nsec; // i32
        		let realsec = diffsec as f64 + diffsub as f64 * 1e-9;
        		Ok(realsec)
    			}),
    			Err(e) => Err(e),
    		}
    }
}

fn main() {
	let num = num_cpus::get();
	let mut single:f64 = 0.0;

	println!("header, thread, speedup efficiency");

	(1..=num as usize).collect::<Vec<_>>().iter().for_each(|&n| {
		let minsec = (1..=10).collect::<Vec<_>>().iter().map(|_|
			match benchmark(n) {
				Ok(realsec) => realsec,
				Err(e) => {println!("error: {}", e); ::std::f64::NAN},
			}
		).fold(0.0/0.0, |m, v| v.min(m));
		match n {
			1 => single = minsec,
			_ => {},
		}
		println!(", {}, {}", n, ((single / minsec / (n as f64)) * 1000.0).round() / 10.0 );
	})
}