use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use rand::prelude::ThreadRng;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::DerefMut;

struct Sampler {
    random: RefCell<ThreadRng>,
    distribution: WeightedIndex<f64>,
    source: Vec<u64>,
}

struct PCSampler {
    random: RefCell<ThreadRng>,
    PCExpringDist: WeightedIndex<u64>,
    source: Vec<(u64, u64)>,
}

impl PCSampler {
    fn new(numPCExpriringArr: Vec<(u64, u64)>)-> PCSampler {
        PCSampler {
            random: RefCell::new(rand::thread_rng()),
            PCExpringDist: WeightedIndex::new(numPCExpriringArr.iter().map(|tuple|tuple.1)).unwrap(),
            source: numPCExpriringArr,
        }
    }

    fn sample(&mut self) -> u64 {
        let indexToDecrement = self.PCExpringDist.sample(self.random.borrow_mut().deref_mut());
        //key that we want to decrement (this is the block that is getting evicted)
        self.source[indexToDecrement].0

    }
}

impl Sampler {
    fn new<T: Iterator<Item = (u64, f64)>>(t: T) -> Sampler {
        let r = RefCell::new(rand::thread_rng());
        let vector: Vec<(u64, f64)> = t.into_iter().collect(); //Guarantees our index ordering.
        let distribution = WeightedIndex::new(vector.iter().map(|(_, weight)| *weight)).unwrap();
        let source = vector.into_iter().map(|(item, _)| item).collect();

        Sampler {
            random: r,
            distribution,
            source,
        }
    }

    fn sample(&self) -> u64 {
        let index = self
            .distribution
            .sample(self.random.borrow_mut().deref_mut());
        self.source[index]
    }
}

struct Simulator {
    VCS: u64,
    PCS: u64,
    //tracks the number of blocks that will expire at a given step in a virtual cache
    VCExpiringBlockTracker: HashMap<u64, u64>,
    //tracks the number of blocks that will expire at a given step in a physical cache
    PCExpiringBlockTracker: HashMap<u64, u64>,
    FERemainingTracker: HashMap<u64, u64>,
    PCSampler: Option<PCSampler>,
    step: u64,
    force_evictions: u64,
}

impl Simulator {
    fn init() -> Simulator {
        Simulator {
            VCS: 0,
            PCS: 0,
            VCExpiringBlockTracker: HashMap::new(),
            PCExpiringBlockTracker: HashMap::new(),
            FERemainingTracker: HashMap::new(),
            PCSampler: None,
            step: 0,
            force_evictions: 0,
        }
    }

    fn add_tenancy(&mut self, tenancy: u64, fixed: u64) {
        self.update(fixed);
        self.VCS += 1;
        self.PCS += 1;
        let target = tenancy + self.step;
        let VC_expirations_at_step = self.VCExpiringBlockTracker.get(&target).copied().unwrap_or(0);
        let PC_expirations_at_step = self.PCExpiringBlockTracker.get(&target).copied().unwrap_or(0);
        self.VCExpiringBlockTracker.insert(target, VC_expirations_at_step + 1);
        self.PCExpiringBlockTracker.insert(target, PC_expirations_at_step + 1);
    }

    fn update(&mut self, fixed: u64) {
        self.step += 1;
        let numPCExpiring = self.PCExpiringBlockTracker.remove(&self.step).unwrap_or(0);
        // print!("num expiring {:#?}  PCS {:#?} ", numPCExpiring, self.PCS);
        self.VCS -= self.VCExpiringBlockTracker.remove(&self.step).unwrap_or(0);
        self.PCS -= numPCExpiring;
        //FE needed if one block doesn't expire since there is a new tenancy added at every step also must need PCS < than the cache size
        if numPCExpiring < 1 && self.PCS > fixed {
            // println!("force exicting\n");
            self.force_evcit();
            self.force_evictions += 1;
            self.PCS -= 1;
        }
    }

    fn force_evcit(&mut self) {
        self.PCSampler = Some(PCSampler::new(self.PCExpiringBlockTracker.iter().map(|(key, value)| (*key, *value)).collect()));
        let keyToDecrement = self.PCSampler.as_mut().unwrap().sample();
        self.PCExpiringBlockTracker.insert(keyToDecrement, self.PCExpiringBlockTracker.get(&keyToDecrement).unwrap() - 1);
        self.FERemainingTracker.insert(keyToDecrement - self.step, self.FERemainingTracker.get(&(keyToDecrement- self.step)).unwrap_or(&0) + 1);
    }

    fn get_excess(&self, fixed: u64) -> u64 {
        if self.VCS <= fixed {
            0
        } else {
            self.VCS - fixed
        }
    }

    fn _get_size(&self) -> u64 {
        self.VCS
    }
}

fn caching(ten_dist: Sampler, cache_size: u64, delta: f64) -> (u64, u64, u64, HashMap<u64, u64>) {
    let mut cache = Simulator::init();
    let mut trace_len: u64 = 0;
    let mut samples_to_issue: u64 = 1024;
    let mut prev_output: Option<f64> = None;
    let mut total_overalloc: u64 = 0;
    loop {
        for _ in 0..samples_to_issue {
            trace_len += 1;
            let tenancy = ten_dist.sample();
            cache.add_tenancy(tenancy, cache_size);
            total_overalloc += cache.get_excess(cache_size);
        }
        if prev_output.is_some()
            && ((total_overalloc as f64) / (trace_len as f64) - prev_output.unwrap()) < delta
        {
            return (total_overalloc, trace_len, cache.force_evictions, cache.FERemainingTracker.clone());
        }
        prev_output = Some((total_overalloc as f64) / (trace_len as f64));
        samples_to_issue *= 2;
    }
}

fn main() {
    let data = vec![
        (1, 0.333),
        (5, 0.333),
        (15, 0.333),
    ];
    // let x = vec![1, 2, 3];
    // let y = vec![1.0, 2.0, 3.0];
    // let iterator = x.into_iter().zip(y.into_iter());
    let (over_alloc, trace_len, forced_evictions, FE_remaining_dist) = caching(Sampler::new(data.into_iter()), 10, 0.05);

    println!(
        "over_alloc: {}, trace_len: {}, div : {}, forced_evictions : {}, \nFE_remaining_dist : {:#?}",
        over_alloc,
        trace_len,
        over_alloc as f64 / trace_len as f64,
        forced_evictions,
        FE_remaining_dist,
    );
}
