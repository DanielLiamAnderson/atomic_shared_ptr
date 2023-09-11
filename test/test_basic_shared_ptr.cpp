// Tests for the basic shared_ptr.
//

#include "gtest/gtest.h"

#include <parlay/basic_atomic_shared_ptr.hpp>


TEST(TestSharedPtr, TestDefaultConstruction) {
  parlay::basic::shared_ptr<int> s;
}


TEST(TestSharedPtr, TestMoveConstructor) {
  parlay::basic::shared_ptr<int> src(new int(1729));

  ASSERT_TRUE(src);
  ASSERT_EQ(*src, 1729);

  parlay::basic::shared_ptr<int> dest(std::move(src));
  
  ASSERT_FALSE(src);
  ASSERT_TRUE(dest);
  ASSERT_EQ(*dest, 1729);
}

TEST(TestSharedPtr, TestMoveAssign) {
  parlay::basic::shared_ptr<int> src(new int(123));
  parlay::basic::shared_ptr<int> dest(new int(888));

  ASSERT_TRUE(src);
  ASSERT_EQ(*src, 123);

  ASSERT_TRUE(dest);
  ASSERT_EQ(*dest, 888);

  dest = std::move(src);

  ASSERT_FALSE(src);
  ASSERT_TRUE(dest);
  ASSERT_EQ(*dest, 123);
}
