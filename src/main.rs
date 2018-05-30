extern crate time;

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

fn benchmark() {
    let mut x = (0..NUM_VALUES as i64).collect::<Vec<_>>();
    for xi in &mut x[1..] {
        *xi = logisticsmap_loop_calc(LOOP, *xi, P, MU)
    }
    println!("1: {}, 10000: {}", x[1], x[10000]);
}

fn main() {
    let start_time = time::get_time();
    benchmark();
    let end_time = time::get_time();
    let diffsec = end_time.sec - start_time.sec;   // i64
    let diffsub = end_time.nsec - start_time.nsec; // i32
    let realsec = diffsec as f64 + diffsub as f64 * 1e-9;
    println!("{:.6} sec", realsec);
}
