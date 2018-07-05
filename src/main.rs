extern crate time;
extern crate ocl;
extern crate rayon;

use ocl::{ProQue, Buffer, MemFlags};
use rayon::prelude::*;

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

fn benchmark_cpu() {
    let r1 = (0..NUM_VALUES as i64).collect::<Vec<_>>().par_iter().map(|&x| logisticsmap_loop_calc(LOOP, x, P, MU)).collect::<Vec<i64>>();
    println!("1: {}, 10000: {}", r1[1], r1[10000]);
}

fn benchmark_gpu() {
    let r1 = logistic_map_ocl((0..NUM_VALUES as i64).collect::<Vec<_>>(), P, MU);

    match r1 {
        Ok(r2) => {
            println!("1: {}, 10000: {}", r2[1], r2[10000]);
        },
       Err(_) => {
            println!("error!");
       },
    }
}


fn logistic_map_ocl(x: Vec<i64>, p: i64, mu: i64) -> ocl::Result<(Vec<i64>)> {
    let src = r#"
        __kernel void calc(__global long* input, __global long* output, long p, long mu) {
            size_t i = get_global_id(0);
            long x = input[i];
            x = mu * x * (x + 1) % p;
            x = mu * x * (x + 1) % p;
            x = mu * x * (x + 1) % p;
            x = mu * x * (x + 1) % p;
            x = mu * x * (x + 1) % p;
            x = mu * x * (x + 1) % p;
            x = mu * x * (x + 1) % p;
            x = mu * x * (x + 1) % p;
            x = mu * x * (x + 1) % p;
            x = mu * x * (x + 1) % p;
            output[i] = x;
        }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .dims(x.len())
        .build().expect("Build ProQue");

    let source_buffer = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(x.len())
        .copy_host_slice(&x)
        .build()?;

    let result_buffer: Buffer<i64> = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(x.len())
        .build()?;

    let kernel = pro_que.kernel_builder("calc")
        .arg(&source_buffer)
        .arg(&result_buffer)
        .arg(p)
        .arg(mu)
        .build()?;

    unsafe { kernel.enq()?; }

    let mut vec_result = vec![0; result_buffer.len()];
    result_buffer.read(&mut vec_result).enq()?;
    Ok(vec_result)
}


fn main() {
    {
        let start_time = time::get_time();
        benchmark_gpu();
        let end_time = time::get_time();
        let diffsec = end_time.sec - start_time.sec;   // i64
        let diffsub = end_time.nsec - start_time.nsec; // i32
        let realsec = diffsec as f64 + diffsub as f64 * 1e-9;
        println!("GPU: {:.6} sec", realsec);
    }
    {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(32).build_global().unwrap();
        let start_time = time::get_time();
        benchmark_cpu();
        let end_time = time::get_time();
        let diffsec = end_time.sec - start_time.sec;   // i64
        let diffsub = end_time.nsec - start_time.nsec; // i32
        let realsec = diffsec as f64 + diffsub as f64 * 1e-9;
        println!("CPU: {:.6} sec", realsec);        
    }
}
