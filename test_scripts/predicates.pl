% Composite Predicates for Nations KG in ProbLog (compatible with SDD)

% 1. Conflict Intensity: accusations and embassy attacks indicate high tension
% conflict_intensity(A, B) :- accusation(A, B), attackembassy(A, B).
conflict_intensity(A, B) :- accusation(A, B), attackembassy(A, B).

% 2. Hostile Exchange: reciprocal accusations or aidenemy both ways
% hostile_exchange(A, B) :- accusation(A, B), accusation(B, A).
% hostile_exchange(A, B) :- aidenemy(A, B), aidenemy(B, A).
hostile_exchange(A, B) :- accusation(A, B), accusation(B, A).
hostile_exchange(A, B) :- aidenemy(A, B), aidenemy(B, A).

% 3. Diplomatic Siege: boycottembargo or blockpositionindex
% diplomatic_siege(A, B) :- boycottembargo(A, B).
% diplomatic_siege(A, B) :- blockpositionindex(A, B).
diplomatic_siege(A, B) :- boycottembargo(A, B).
diplomatic_siege(A, B) :- blockpositionindex(A, B).

% 4. Cultural Bridge: shared book translations or NGO cooperation
% cultural_bridge(A, B) :- booktranslations(A, B).
% cultural_bridge(A, B) :- ngo(A, B).
cultural_bridge(A, B) :- booktranslations(A, B).
cultural_bridge(A, B) :- ngo(A, B).

% 5. Military Alignment: military alliance or negativecomm (negative communications)
% military_alignment(A, B) :- militaryalliance(A, B).
% military_alignment(A, B) :- negativecomm(A, B).
military_alignment(A, B) :- militaryalliance(A, B).
military_alignment(A, B) :- negativecomm(A, B).

% 6. Humanitarian Concern: economicaid or releconomicaid
% humanitarian_concern(A, B) :- economicaid(A, B).
% humanitarian_concern(A, B) :- releconomicaid(A, B).
humanitarian_concern(A, B) :- economicaid(A, B).
humanitarian_concern(A, B) :- releconomicaid(A, B).

% 7. Shared Indoctrination: conferences and student exchanges
% shared_indoctrination(A, B) :- conferences(A, B).
% shared_indoctrination(A, B) :- students(A, B).
shared_indoctrination(A, B) :- conferences(A, B).
shared_indoctrination(A, B) :- students(A, B).
