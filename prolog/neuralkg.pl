:- module(neuralkg, [test_prolog_python/2, query_relation/3, query_pandas/3, test_wfs/1]).
:- use_module(library(janus)).

% Simple test that works with pure Prolog using correct py_call syntax
test_prolog_python(Arg1, Result) :-
    % Get Python's sys.version using proper colon notation
    py_call(sys:version, Version),
    atomic_list_concat(['Echo from Prolog: ', Arg1, ' (Python: ', Version, ')'], Result).

% Simple static query relation
query_relation(Relation, Args, Results) :-
    % Convert Args to string for display
    term_string(Args, ArgsStr),
    atomic_list_concat(['Queried ', Relation, ' with args ', ArgsStr], Results).

% Bidirectional test: Prolog predicate that queries Python for dataframe information
% Uses the direct module:function format to avoid variable instantiation issues
query_pandas(Predicate, Constraints, Results) :-
    % Make sure all arguments are bound/ground
    atom(Predicate),  % Ensure predicate name is an atom
    ground(Constraints),  % Ensure constraints are ground terms

    % Call Python function directly without getting the function reference first
    py_call('neuralkg.prolog.bridge':query_dataframe(Predicate, Constraints), PyResults),
    
    % Convert Python results to Prolog results
    (PyResults = [] -> 
        Results = 'No matching records found'
    ; 
        term_string(PyResults, ResultsStr),
        atomic_list_concat(['Found records: ', ResultsStr], Results)
    ).

% Simple well-founded semantics test
test_wfs(X) :-
    member(X, [true, false, undefined]). % Add undefined as a possible result
