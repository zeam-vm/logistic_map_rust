#[macro_use] extern crate lazy_static;

extern crate time;
extern crate rayon;

use rayon::prelude::*;
use rayon::ThreadPool;

const LOOP: usize = 10;
const P: i64 = 6_700_417;
const MU: i64 = 22;
const NUM_VALUES: usize = 0x2_000_000;

lazy_static! {
    static ref _THREAD_POOL: ThreadPool = rayon::ThreadPoolBuilder::new().num_threads(32).build().unwrap();
}

fn logisticsmap_calc(x: i64, p: i64, mu: i64) -> i64 {
    mu * x * (x + 1) % p
}

fn logisticsmap_loop_calc(num: usize, x: i64, p: i64, mu: i64) -> i64 {
    (0..num).fold(x, |x, _| logisticsmap_calc(x, p, mu))
}

fn benchmark_cpu_single() {
    let r1 = (0..NUM_VALUES as i64).collect::<Vec<_>>().iter().map(|&x| logisticsmap_loop_calc(LOOP, x, P, MU)).collect::<Vec<i64>>();
    println!("1: {}, 10000: {}", r1[1], r1[10000]);
}

fn benchmark_cpu_multi() {
    let r1 = (0..NUM_VALUES as i64).collect::<Vec<_>>().par_iter().map(|&x| logisticsmap_loop_calc(LOOP, x, P, MU)).collect::<Vec<i64>>();
    println!("1: {}, 10000: {}", r1[1], r1[10000]);
}

fn main() {
    {
        let start_time = time::get_time();
        benchmark_cpu_single();
        let end_time = time::get_time();
        let diffsec = end_time.sec - start_time.sec;   // i64
        let diffsub = end_time.nsec - start_time.nsec; // i32
        let realsec = diffsec as f64 + diffsub as f64 * 1e-9;
        println!("CPU(1): {:.6} sec", realsec);
    }
    {
        let start_time = time::get_time();
        benchmark_cpu_multi();
        let end_time = time::get_time();
        let diffsec = end_time.sec - start_time.sec;   // i64
        let diffsub = end_time.nsec - start_time.nsec; // i32
        let realsec = diffsec as f64 + diffsub as f64 * 1e-9;
        println!("CPU(m): {:.6} sec", realsec);
    }
}
