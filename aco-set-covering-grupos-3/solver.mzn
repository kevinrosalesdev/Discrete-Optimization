% Use this editor as a MiniZinc scratch book
int: ITEM_COUNT;
int: NSET_COUNT;

set of int: NODES = 0..ITEM_COUNT-1;
set of int: SET_COUNT = 1..NSET_COUNT;

array[SET_COUNT] of var set of NODES: sets;
array[SET_COUNT] of int: cost;
array[SET_COUNT] of var 0..1: selected_ones;

constraint forall(node in NODES)(sum(nset in SET_COUNT)
(
    selected_ones[nset]*(node in sets[nset])) >= 1
);

var int: totalCost = sum(i in SET_COUNT)(cost[i] * selected_ones[i]);
solve minimize(totalCost);