// Tests for the custom shared_ptr.
//
// Some of these were stolen from the Microsoft STL Github.

#include "gtest/gtest.h"

#include <parlay/shared_ptr.hpp>


class Base {
public:
    Base() {}
    virtual ~Base() {}

    virtual std::string str() const {
        return "Base";
    }

private:
    Base(const Base&);
    Base& operator=(const Base&);
};

class Derived : public Base {
public:
    virtual std::string str() const {
        return "Derived";
    }
};




TEST(TestSharedPtr, TestDefaultConstruction) {
  parlay::shared_ptr<int> s;
}


TEST(TestSharedPtr, TestMoveConstructor) {
  parlay::shared_ptr<int> src(new int(1729));

  ASSERT_TRUE(src);
  ASSERT_EQ(*src, 1729);

  parlay::shared_ptr<int> dest(std::move(src));
  
  ASSERT_FALSE(src);
  ASSERT_TRUE(dest);
  ASSERT_EQ(*dest, 1729);
}

TEST(TestSharedPtr, TestMoveAssign) {
  parlay::shared_ptr<int> src(new int(123));
  parlay::shared_ptr<int> dest(new int(888));

  ASSERT_TRUE(src);
  ASSERT_EQ(*src, 123);

  ASSERT_TRUE(dest);
  ASSERT_EQ(*dest, 888);

  dest = std::move(src);

  ASSERT_FALSE(src);
  ASSERT_TRUE(dest);
  ASSERT_EQ(*dest, 123);
}

TEST(TestSharedPtr, TestAliasMoveConstructor) {
  parlay::shared_ptr<Derived> src(new Derived);

  ASSERT_TRUE(src);
  ASSERT_EQ(src->str(), "Derived");

  parlay::shared_ptr<Base> dest(std::move(src));

  ASSERT_FALSE(src);
  ASSERT_TRUE(dest);
  ASSERT_EQ(dest->str(), "Derived");
}

