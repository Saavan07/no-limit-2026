// Exact sample scores for every allocation using prefix sums on a simplex.
// The original scorer remains unmodified and supplies the game payoff table.
#include "../data/parry-replay/lookup.cpp"
#include <array>
#include <chrono>
#include <limits>

namespace {
std::once_flag differences_initialized;
std::array<std::array<uint8_t,5>,states> ternary;
struct Term {uint8_t state; int16_t value;};
std::vector<std::vector<Term>> differences;
void init_differences() {
  std::call_once(initialized,init);
  for(int s=0;s<states;++s) {int t=s;for(int j=0;j<5;++j){ternary[s][j]=t%3;t/=3;}}
  differences.resize(patterns);
  for(int code=0;code<patterns;++code) {
    int d[states];for(int s=0;s<states;++s)d[s]=payout[code*states+s]/3;
    for(int power=1;power<states;power*=3)
      for(int s=states-1;s>=0;--s)if((s/power)%3)d[s]-=d[s-power];
    for(int s=0;s<states;++s)if(d[s])differences[code].push_back({(uint8_t)s,(int16_t)d[s]});
  }
}
struct SimplexIndex {
  uint64_t choose[106][6]{};
  explicit SimplexIndex() {
    for(int n=0;n<=105;++n) {choose[n][0]=1;for(int k=1;k<=5;++k)choose[n][k]=(n?choose[n-1][k-1]+choose[n-1][k]:0);}
  }
  inline size_t address(int B,int a,int b,int c,int d,int e=0)const {
    int r=B;uint64_t index=choose[r+5][5]-choose[r-a+5][5];r-=a;
    index+=choose[r+4][4]-choose[r-b+4][4];r-=b;
    index+=choose[r+3][3]-choose[r-c+3][3];r-=c;
    index+=choose[r+2][2]-choose[r-d+2][2];
    return (size_t)(index+e);
  }
};
inline void add_line(int64_t* __restrict dst,const int64_t* __restrict src,int length) {
  for(int i=0;i<length;++i)dst[i]+=src[i];
}
template<int Axis> void prefix_axis(std::vector<int64_t>& grid,int B,const SimplexIndex& index) {
  size_t cursor=0;
  for(int a=0;a<=B;++a)for(int b=0;b<=B-a;++b)
    for(int c=0;c<=B-a-b;++c)for(int d=0;d<=B-a-b-c;++d) {
      const int length=B-a-b-c-d+1;
      if constexpr(Axis==4) {
        for(int e=1;e<length;++e)grid[cursor+e]+=grid[cursor+e-1];
      } else {
        const int coordinate=(Axis==0?a:Axis==1?b:Axis==2?c:d);
        if(coordinate) {
          const size_t previous=index.address(B,a-(Axis==0),b-(Axis==1),c-(Axis==2),d-(Axis==3));
          add_line(grid.data()+cursor,grid.data()+previous,length);
        }
      }
      cursor+=length;
    }
}
}
extern "C" {
int simplex_sample_sums(int B,const uint8_t* maxima,const int32_t* codes,
                        const int64_t* counts,int groups,int64_t* out,double* timing) {
  if(B<0||B>100||groups<=0)return -1;
  using Clock=std::chrono::steady_clock;const auto start=Clock::now();
  std::call_once(differences_initialized,init_differences);
  const SimplexIndex index;
  try {
    std::vector<int64_t> grid(index.choose[B+5][5],0);
    int64_t total=0;
    for(int g=0;g<groups;++g) {
      if(codes[g]<0||codes[g]>=patterns||counts[g]<=0)return -2;
      total+=counts[g];const uint8_t* m=maxima+5*g;
      for(const auto& term:differences[codes[g]]) {
        const auto& s=ternary[term.state];int x[5],spent=0;
        for(int j=0;j<5;++j){x[j]=s[j]?m[j]+s[j]-1:0;spent+=x[j];}
        if(spent<=B)grid[index.address(B,x[0],x[1],x[2],x[3],x[4])]+=counts[g]*term.value;
      }
    }
    const auto scattered=Clock::now();
    prefix_axis<0>(grid,B,index);prefix_axis<1>(grid,B,index);prefix_axis<2>(grid,B,index);
    prefix_axis<3>(grid,B,index);prefix_axis<4>(grid,B,index);
    const auto prefixed=Clock::now();size_t position=0;
    for(int a=0;a<=B;++a)for(int b=0;b<=B-a;++b)
      for(int c=0;c<=B-a-b;++c)for(int d=0;d<=B-a-b-c;++d) {
        const int e=B-a-b-c-d;const int64_t value=grid[index.address(B,a,b,c,d,e)];
        if(value<0||value>36*total)return -3;
        out[position++]=value;
      }
    const auto done=Clock::now();
    timing[0]=std::chrono::duration<double>(scattered-start).count();
    timing[1]=std::chrono::duration<double>(prefixed-scattered).count();
    timing[2]=std::chrono::duration<double>(done-prefixed).count();
  }catch(const std::bad_alloc&){return -4;}
  return 0;
}
void direct_sample_sums(const uint8_t* allocations,int n,const uint8_t* maxima,
                       const int32_t* codes,const int64_t* counts,int groups,int64_t* out,int workers) {
  std::call_once(initialized,init);
  auto rows=[&](int first,int last){
    for(int i=first;i<last;++i){int64_t total=0;for(int g=0;g<groups;++g)
      total+=counts[g]*(payout[index_for(allocations+5*i,maxima+5*g,codes[g])]/3);out[i]=total;}
  };
  workers=std::max(1,std::min(workers,n));
  if(workers==1){rows(0,n);return;}
  std::vector<std::thread> threads;
  for(int w=0;w<workers;++w)threads.emplace_back(rows,(int)((int64_t)n*w/workers),(int)((int64_t)n*(w+1)/workers));
  for(auto& thread:threads)thread.join();
}
void direct_sample_matrix4(const uint8_t* allocations,int n,const uint8_t* maxima,
                          const int32_t* codes,int groups,uint8_t* out,int workers) {
  std::call_once(initialized,init);
  auto rows=[&](int first,int last){for(int i=first;i<last;++i)for(int g=0;g<groups;++g)
    out[(int64_t)i*groups+g]=payout[index_for(allocations+5*i,maxima+5*g,codes[g])]/3;};
  workers=std::max(1,std::min(workers,n));
  if(workers==1){rows(0,n);return;}
  std::vector<std::thread> threads;
  for(int w=0;w<workers;++w)threads.emplace_back(rows,(int)((int64_t)n*w/workers),(int)((int64_t)n*(w+1)/workers));
  for(auto& thread:threads)thread.join();
}
}
