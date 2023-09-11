#include <cstdlib>

#include <latch>
#include <numeric>
#include <thread>
#include <vector>


TEST(TestAtomicSharedPtr, TestConstructionEmpty) {
  atomic_shared_ptr<int> p;
  
  auto s = p.load();
  ASSERT_FALSE(s);
  ASSERT_EQ(s, nullptr);
}

TEST(TestAtomicSharedPtr, TestConstructionValue) {
  shared_ptr<int> s{new int(5)};
  atomic_shared_ptr<int> p{std::move(s)};
  
  auto s2 = p.load();
  ASSERT_EQ(s2.use_count(), 2);
  ASSERT_EQ(*s2, 5);
}

TEST(TestAtomicSharedPtr, TestStoreCopy) {
  atomic_shared_ptr<int> p;
  
  shared_ptr<int> s{new int(5)};
  ASSERT_EQ(s.use_count(), 1);
  p.store(s);
  ASSERT_EQ(s.use_count(), 2);
  
  auto s2 = p.load();
  ASSERT_EQ(s2.use_count(), 3);
  ASSERT_EQ(*s2, 5);
}

TEST(TestAtomicSharedPtr, TestStoreMove) {
  atomic_shared_ptr<int> p;
  
  shared_ptr<int> s{new int(5)};
  auto s2 = s;
  ASSERT_EQ(s.use_count(), 2);
  
  p.store(std::move(s2));
  ASSERT_FALSE(s2);
  ASSERT_EQ(s2, nullptr);
  ASSERT_EQ(s.use_count(), 2);
}

TEST(TestAtomicSharedPtr, TestLoad) {
  shared_ptr<int> s{new int(5)};
  atomic_shared_ptr<int> p{std::move(s)};
  ASSERT_FALSE(s);
  ASSERT_EQ(s, nullptr);
  
  shared_ptr<int> l = p.load();
  ASSERT_EQ(*l, 5);
  ASSERT_EQ(l.use_count(), 2);
}

TEST(TestAtomicSharedPtr, TestExchange) {
  shared_ptr<int> s{new int(5)};
  atomic_shared_ptr<int> p{std::move(s)};
  ASSERT_FALSE(s);
  ASSERT_EQ(s, nullptr);
  
  shared_ptr<int> s2{new int(42)};
  shared_ptr<int> s3 = p.exchange(std::move(s2));
  
  ASSERT_EQ(*s3, 5);
  ASSERT_EQ(s3.use_count(), 1);
  
  shared_ptr<int> l = p.load();
  ASSERT_EQ(*l, 42);
  ASSERT_EQ(l.use_count(), 2);
}

TEST(TestAtomicSharedPtr, TestCompareExchangeWeakTrue) {
  shared_ptr<int> s{new int(5)};
  atomic_shared_ptr<int> p{s};
  ASSERT_TRUE(s);
  ASSERT_EQ(s.use_count(), 2);
  
  shared_ptr<int> s2{new int(42)};
  bool result = p.compare_exchange_weak(s, std::move(s2));
  ASSERT_TRUE(result);
  ASSERT_FALSE(s2);
  ASSERT_EQ(s2, nullptr);
  
  shared_ptr<int> l = p.load();
  ASSERT_EQ(*l, 42);
  ASSERT_EQ(l.use_count(), 2);
}

TEST(TestAtomicSharedPtr, TestCompareExchangeWeakFalse) {
  shared_ptr<int> s{new int(5)};
  atomic_shared_ptr<int> p{s};
  ASSERT_TRUE(s);
  ASSERT_EQ(s.use_count(), 2);
  
  shared_ptr<int> s2{new int(42)};
  shared_ptr<int> s3{new int(5)};
  bool result = p.compare_exchange_weak(s3, std::move(s2));
  ASSERT_FALSE(result);
  
  shared_ptr<int> l = p.load();
  ASSERT_EQ(*l, 5);
  ASSERT_EQ(l.use_count(), 4);
}

TEST(TestAtomicSharedPtr, TestCompareExchangeStrongTrue) {
  shared_ptr<int> s{new int(5)};
  atomic_shared_ptr<int> p{s};
  ASSERT_TRUE(s);
  ASSERT_EQ(s.use_count(), 2);
  
  shared_ptr<int> s2{new int(42)};
  bool result = p.compare_exchange_strong(s, std::move(s2));
  ASSERT_TRUE(result);
  ASSERT_FALSE(s2);
  ASSERT_EQ(s2, nullptr);
  
  shared_ptr<int> l = p.load();
  ASSERT_EQ(*l, 42);
  ASSERT_EQ(l.use_count(), 2);
}

TEST(TestAtomicSharedPtr, TestCompareExchangeStrongFalse) {
  shared_ptr<int> s{new int(5)};
  atomic_shared_ptr<int> p{s};
  ASSERT_TRUE(s);
  ASSERT_EQ(s.use_count(), 2);
  
  shared_ptr<int> s2{new int(42)};
  shared_ptr<int> s3{new int(5)};
  bool result = p.compare_exchange_strong(s3, std::move(s2));
  ASSERT_FALSE(result);
  
  shared_ptr<int> l = p.load();
  ASSERT_EQ(*l, 5);
  ASSERT_EQ(l.use_count(), 4);
}

TEST(TestAtomicSharedPtr, TestConcurrentStoreLoads) {
  
  constexpr std::size_t N = 64;       // Number of threads
  constexpr int M = 10000;            // Number of operations
  
  atomic_shared_ptr<int> s;
  std::latch go{N};

  std::vector<std::jthread> consumers;
  consumers.reserve(N/2);
  std::vector<long long int> consumer_sums(N/2);
  for (size_t i = 0; i < N/2; i++) {
    consumers.emplace_back([i, &s, &consumer_sums, &go] {
      go.arrive_and_wait();
      long long int local_sum = 0;
      for(int j = 0; j < M; j++) {
        auto p = s.load();
        if (p) {
          local_sum += *p;
        }
      }
      consumer_sums[i] = local_sum;
    });
  }

  std::vector<std::jthread> producers;
  producers.reserve(N/2);
  for (size_t i = 0; i < N/2; i++) {
    producers.emplace_back([&s, &go] {
      go.arrive_and_wait();
      for(int j = 0; j < M; j++) {
        s.store(shared_ptr<int>(new int(j)));
      }
    });
  }
}

TEST(TestAtomicSharedPtr, TestConcurrentExchange) {
  
  constexpr std::size_t N = 64;       // Number of threads
  constexpr int M = 10000;            // Number of operations
  
  atomic_shared_ptr<int> s(shared_ptr<int>(new int(0)));
  std::latch go{N};

  std::vector<long long int> local_sums_produced(N);
  std::vector<long long int> local_sums_consumed(N);

  {
    std::vector<std::jthread> threads;
    threads.reserve(N);
    
    for (size_t i = 0; i < N; i++) {
      threads.emplace_back([i, &s, &go, &local_sums_produced, &local_sums_consumed] {
        go.arrive_and_wait();
        long long int local_sum_produced = 0, local_sum_consumed = 0;
        for(int j = 0; j < M; j++) {
          shared_ptr<int> new_sp(new int(std::rand()));
          local_sum_produced += *new_sp;
          shared_ptr<int> old_sp = s.exchange(std::move(new_sp));
          ASSERT_TRUE(old_sp);
          local_sum_consumed += *old_sp;
        }
        local_sums_produced[i] = local_sum_produced;
        local_sums_consumed[i] = local_sum_consumed;
      });
    }
  }   // wait for threads to join

  long long int total_produced = std::accumulate(local_sums_produced.begin(), local_sums_produced.end(), 0LL);
  long long int total_consumed = std::accumulate(local_sums_consumed.begin(), local_sums_consumed.end(), 0LL) + *(s.load());
  
  ASSERT_EQ(total_produced, total_consumed);
}
