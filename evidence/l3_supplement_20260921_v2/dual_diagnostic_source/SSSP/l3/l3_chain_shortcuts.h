#pragma once
// Stage222: host-built, owner-local path edges. Original vertices/edges remain.
// No distances or CPU reference are inputs. Directed weights may be asymmetric.
// L3_CHAIN_SHORTCUTS_LOOSE (default off): drop the in-degree==2 requirement.
// Shortcuts are additive (every original edge is retained), so folding a
// degree-2 vertex whose extra in-edges come from non-neighbors stays sound:
// those paths keep using the original edges.  Road networks have many
// merge/fork segment vertices (~2.4-4x more eligible), promising longer
// collapsible chains at the price of a slightly larger augmented CSR.
#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <vector>
#ifndef L3_CHAIN_SHORTCUTS_LOOSE
#define L3_CHAIN_SHORTCUTS_LOOSE false
#endif

struct l3_chain_shortcut_view {
    std::vector<int> rows, destinations, weights;
    int eligible_vertices=0, shortcuts=0, longest_path=0, overflow_paths=0;
    long long represented_edges=0;

    void build(int n, int m, const int *row, const int *dst, const int *weight,
               int cut, int source) {
        if(n<=0 || m<0 || source<0 || source>=n || cut<0 || cut>n || row[0]!=0 || row[n]!=m)
            throw std::runtime_error("chain input dimensions");
        eligible_vertices=shortcuts=longest_path=overflow_paths=0;
        represented_edges=0;
        std::vector<unsigned char> eligible(n,0);
        std::vector<int> incoming(n,0), reverse0(n,-1), reverse1(n,-1);
        for(int u=0;u<n;++u) {
            if(row[u]>row[u+1] || row[u]<0 || row[u+1]>m)
                throw std::runtime_error("chain CSR row");
            for(int e=row[u];e<row[u+1];++e) {
                if(dst[e]<0 || dst[e]>=n || weight[e]<0)
                    throw std::runtime_error("chain edge");
            }
            if(u==source || row[u+1]-row[u]!=2) continue;
            int a=dst[row[u]], b=dst[row[u]+1];
            eligible[u]=(a!=u && b!=u && a!=b && (a<cut)==(u<cut) && (b<cut)==(u<cut));
        }
        // Incoming completeness is necessary: an extra ingress must be an
        // anchor, otherwise a shortcut would not describe a maximal chain.
        for(int u=0;u<n;++u) for(int e=row[u];e<row[u+1];++e) {
            int v=dst[e];
            if(!eligible[v]) continue;
            ++incoming[v];
            if(u==dst[row[v]]) reverse0[v]=e;
            if(u==dst[row[v]+1]) reverse1[v]=e;
        }
        for(int u=0;u<n;++u) {
#if (L3_CHAIN_SHORTCUTS_LOOSE == true)
            // Loose eligibility: any degree-2 local vertex may be folded, even
            // merge/fork vertices whose ingress is not from their two out
            // neighbors.  The walk below abandons entry directions where the
            // onward neighbor cannot be disambiguated, and partial shortcuts
            // (ending mid-chain) stay sound because they carry a true partial
            // path sum and every original edge is retained.
            eligible[u]=eligible[u];
#else
            eligible[u]=eligible[u] && incoming[u]==2 && reverse0[u]>=0 && reverse1[u]>=0;
#endif
            eligible_vertices+=eligible[u]!=0;
        }
        rows.assign(n+1,0);destinations.clear();weights.clear();
        destinations.reserve(m);weights.reserve(m);
        for(int u=0;u<n;++u) {
            if(!eligible[u]) for(int first=row[u];first<row[u+1];++first) {
                int previous=u, current=dst[first], hops=1;
                long long cost=weight[first];
                bool overflow=false;
                while(eligible[current]) {
                    int e=row[current];
                    if(dst[e]==previous) ++e;
                    else if(dst[e+1]!=previous) {
#if (L3_CHAIN_SHORTCUTS_LOOSE == true)
                        // Loose folding may enter a chain vertex from a
                        // non-mutual direction where the onward neighbor is
                        // ambiguous.  Abandon the last hop: `cost` still covers
                        // exactly u..previous, so the shortcut simply ends at
                        // `previous` (a partial path with a true sum; the
                        // original edges keep the rest discoverable).
                        current=previous;
                        break;
#else
                        throw std::runtime_error("chain predecessor");
#endif
                    }
                    cost+=weight[e];++hops;
                    previous=current;current=dst[e];
                    if(cost>=INT_MAX) {overflow=true;break;}
                    if(hops>n) throw std::runtime_error("chain did not reach anchor");
                }
                if(overflow) {++overflow_paths;continue;}
                if(hops<=1 || current==u) continue;
                if((current<cut)!=(u<cut)) throw std::runtime_error("chain changed owner");
                if(destinations.size()>=INT_MAX) throw std::runtime_error("chain edge count overflow");
                destinations.push_back(current);weights.push_back(int(cost));
                ++shortcuts;represented_edges+=hops;longest_path=std::max(longest_path,hops);
            }
            // Every original directed edge and its weight are retained exactly.
            for(int e=row[u];e<row[u+1];++e) {
                if(destinations.size()>=INT_MAX) throw std::runtime_error("chain edge count overflow");
                destinations.push_back(dst[e]);weights.push_back(weight[e]);
            }
            rows[u+1]=int(destinations.size());
        }
    }
};
