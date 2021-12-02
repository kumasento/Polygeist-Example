template <typename... Args> void foo(Args... args);

int main() {
  foo(1);
  foo(1, 2);
  foo(1, 2, 3);
}
