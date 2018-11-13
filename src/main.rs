extern crate time;
extern crate rayon;
extern crate num_cpus;
extern crate ocl;

use ocl::{ProQue, Buffer, MemFlags};
use ocl::{Platform, Device};
use ocl::enums::{PlatformInfo, DeviceInfo, DeviceInfoResult};
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

fn work_dim(device: &Device) -> u32 {
	match device.info(DeviceInfo::MaxWorkItemDimensions) {
		Ok(DeviceInfoResult::MaxWorkItemDimensions(res)) => res,
    _ => {
      println!("failed to get DeviceInfoResult::MaxWorkItemDimensions");
      1
    },
  }
}

fn max_work_group_size(device: &Device) -> usize {
	match device.info(DeviceInfo::MaxWorkGroupSize) {
		Ok(DeviceInfoResult::MaxWorkGroupSize(res)) => res,
		_ => {
			println!("failed to get DeviceInfoResult::MaxWorkGroupSize");
			1
    },
  }
}

fn compute_unit_num(device: &Device) -> u32 {
	match device.info(DeviceInfo::MaxComputeUnits) {
		Ok(DeviceInfoResult::MaxComputeUnits(res)) => res,
    _ => {
      println!("failed to get DeviceInfoResult::MaxComputeUnits");
      0
    },
  }
}

fn max_local_memory_size(device: &Device) -> u32 {
	match device.info(DeviceInfo::LocalMemSize) {
		Ok(DeviceInfoResult::LocalMemSize(res)) => res as u32,
    _ => {
      println!("failed to get DeviceInfoResult::LocalMemSize");
      0
    },
  }
}

fn max_work_item_size(device: &Device) -> Vec<usize> {
	match device.info(DeviceInfo::MaxWorkItemSizes) {
		Ok(DeviceInfoResult::MaxWorkItemSizes(res)) => res,
    _ => {
      println!("failed to get DeviceInfoResult::MaxWorkGroupSize");
      vec![1; 3]
    },
  }
}

fn benchmark_gpu() -> i64 {
    let r1 = logistic_map_ocl((0..NUM_VALUES as i64).collect::<Vec<_>>(), P, MU);

    match r1 {
        Ok(r2) => {
            r2[10000]
        },
       Err(_) => {
            println!("error!");
            0
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

pub fn gpuinfo() {
  let platforms = Platform::list();

  println!("platform num:{:?}", platforms.len());

  for p_idx in 0..platforms.len() {
  	let platform = &platforms[p_idx];
  	let devices = Device::list_all(platform).unwrap();
  	if devices.is_empty() { continue; }
  	println!("Profile:{:?}", platform.info(PlatformInfo::Profile));
  	println!("Version:{:?}", platform.info(PlatformInfo::Version));
  	println!("Name:{:?}", platform.info(PlatformInfo::Name));
  	println!("Vendor:{:?}", platform.info(PlatformInfo::Vendor));
  	for device in devices.iter() {
  		println!("----");
  		println!("Device Name:{:?}", device.name().unwrap());
  		println!("Device Vendor:{:?}", device.vendor().unwrap());
  		println!("  work_dim: {:?}", work_dim(device));
  		println!("  compute_unit_num: {:?}", compute_unit_num(device));
  		println!("  max_work_group_size: {:?}", max_work_group_size(device));
  		println!("  max_local_memory_size: {:?}", max_local_memory_size(device));
  		// max_local_memory_size を超えたデータを
  		// ローカルメモリ(__local修飾子)としてカーネルに渡すと
  		// エラーが発生する．
  		println!("  max_work_item_size: {:?}", max_work_item_size(device));
  	}
  }
}

fn main() {
	{
		gpuinfo();
  	let minsec = (1..=10).collect::<Vec<_>>().iter().map(|_|
  	{
    	let start_time = time::get_time();
    	benchmark_gpu();
    	let end_time = time::get_time();
    	let diffsec = end_time.sec - start_time.sec;   // i64
    	let diffsub = end_time.nsec - start_time.nsec; // i32
    	let realsec = diffsec as f64 + diffsub as f64 * 1e-9;
    	realsec
  	}).fold(0.0/0.0, |m, v| v.min(m));
  	println!("GPU: {}", (minsec * 1000.0).round() / 1000.0);
	}

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
	});

}