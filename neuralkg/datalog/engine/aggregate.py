# Aggregate function registry and helpers

AGGREGATE_FUNCS = {}

def agg_sum(values):
    return sum(values)

def agg_count(values):
    return len(values)

def agg_max(values):
    return max(values)

def agg_min(values):
    return min(values)

def agg_avg(values):
    return sum(values) / len(values) if values else None

def init_aggregate_registry():
    AGGREGATE_FUNCS['agg_sum'] = agg_sum
    AGGREGATE_FUNCS['agg_count'] = agg_count
    AGGREGATE_FUNCS['agg_max'] = agg_max
    AGGREGATE_FUNCS['agg_min'] = agg_min
    AGGREGATE_FUNCS['agg_avg'] = agg_avg
