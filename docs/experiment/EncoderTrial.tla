--------------------------- MODULE EncoderTrial ---------------------------
EXTENDS Integers, FiniteSets
CONSTANTS DevReports, TestReports, MaxTrials, Patience
Kinds == 1..4
Arms == {"ref","en","ko"}
Students == {"en","ko"}
N == 48
Total == 4*N
Sum4(v) == v[1] + v[2] + v[3] + v[4]
Abs(x) == IF x < 0 THEN -x ELSE x

Valid(m) ==
  /\ m.hits \in [Arms -> [Kinds -> 0..N]]
  /\ m.usable \in [Arms -> 0..Total]
  /\ m.peak \in [Arms -> Nat \ {0}]
  /\ m.p95 \in [Arms -> Nat \ {0}]
  /\ m.evidence \in BOOLEAN
  /\ m.standalone \in BOOLEAN
  /\ m.exports \in BOOLEAN

Intent(m,a) ==
  /\ 100*Sum4(m.hits[a]) >= 80*Total
  /\ 100*(Sum4(m.hits[a])-Sum4(m.hits["ref"])) >= -5*Total
  /\ \A k \in Kinds :
       /\ 100*m.hits[a][k] >= 70*N
       /\ 100*(m.hits[a][k]-m.hits["ref"][k]) >= -10*N

LanguageGap(m) ==
  100*Abs(Sum4(m.hits["ko"])-Sum4(m.hits["en"])) <= 5*Total
Quality(m,a) ==
  /\ 100*m.usable[a] >= 90*Total
  /\ 100*(m.usable[a]-m.usable["ref"]) >= -5*Total
Cost(m,a) ==
  /\ 100*m.peak[a] <= 80*m.peak["ref"]
  /\ 100*m.p95[a] <= 120*m.p95["ref"]

Gate(m) ==
  /\ Valid(m) /\ m.evidence /\ m.standalone /\ m.exports
  /\ LanguageGap(m)
  /\ \A a \in Students : Intent(m,a) /\ Quality(m,a) /\ Cost(m,a)

PassedChecks(m) == Cardinality({i \in 1..7 :
  CASE i=1 -> Intent(m,"en") [] i=2 -> Intent(m,"ko")
    [] i=3 -> Quality(m,"en") [] i=4 -> Quality(m,"ko")
    [] i=5 -> Cost(m,"en") [] i=6 -> Cost(m,"ko") [] OTHER -> LanguageGap(m)})
Score(m) == PassedChecks(m)*(2*Total+1) + Sum4(m.hits["en"])+Sum4(m.hits["ko"])

VARIABLES phase, trial, best, stale, locked, tests, last
v == <<phase, trial, best, stale, locked, tests, last>>
Init == /\ phase="Dev" /\ trial=0 /\ best=-1 /\ stale=0
        /\ locked=0 /\ tests=0 /\ last="none"

Evaluate(m) ==
  /\ phase="Dev" /\ trial < MaxTrials /\ m \in DevReports
  /\ trial'=trial+1 /\ UNCHANGED <<tests,last>>
  /\ IF ~(Valid(m) /\ m.evidence) THEN
        /\ phase'="Blocked" /\ UNCHANGED <<best,stale,locked>>
     ELSE LET s == Score(m)
              newer == s > best
              n == IF newer THEN 0 ELSE stale+1
          IN /\ best'=IF newer THEN s ELSE best
             /\ stale'=n
             /\ locked'=IF Gate(m) THEN trial+1 ELSE 0
             /\ phase'=IF Gate(m) THEN "Frozen"
                       ELSE IF trial+1=MaxTrials \/ n>=Patience THEN "Stopped"
                       ELSE "Dev"

Confirm(m) ==
  /\ phase="Frozen" /\ tests=0 /\ m \in TestReports
  /\ tests'=1 /\ last'=m /\ UNCHANGED <<trial,best,stale,locked>>
  /\ phase'=IF ~(Valid(m) /\ m.evidence) THEN "Blocked"
            ELSE IF Gate(m) THEN "Accepted" ELSE "Rejected"

Terminal == phase \in {"Accepted","Rejected","Stopped","Blocked"}
Next == (\E m \in DevReports : Evaluate(m))
        \/ (\E m \in TestReports : Confirm(m)) \/ (Terminal /\ UNCHANGED v)
Spec == Init /\ [][Next]_v /\ WF_v(Next)
Bounds == trial \in 0..MaxTrials /\ tests \in 0..1 /\ stale \in 0..Patience
FrozenOnly == (phase="Frozen" \/ tests=1) => (locked=trial /\ locked>0)
AcceptedSound == IF phase="Accepted" THEN tests=1 /\ Gate(last) ELSE TRUE
Terminates == <>Terminal

\* Finite report examples used only by TLC.
Uniform(h) == [a \in Arms |-> [k \in Kinds |-> h]]
Good == [hits |-> Uniform(42), usable |-> [a \in Arms |-> 181],
         peak |-> [a \in Arms |-> IF a="ref" THEN 20000 ELSE 16000],
         p95 |-> [a \in Arms |-> IF a="ref" THEN 1000 ELSE 1200],
         evidence |-> TRUE, standalone |-> TRUE, exports |-> TRUE]
Examples == {Good, [Good EXCEPT !.evidence=FALSE], [Good EXCEPT !.exports=FALSE],
             [Good EXCEPT !.standalone=FALSE], [Good EXCEPT !.p95["en"]=1201],
             [Good EXCEPT !.peak["ko"]=16001], [Good EXCEPT !.hits["ko"][1]=37],
             [Good EXCEPT !.hits["ko"]=[k \in Kinds |-> 39]],
             [Good EXCEPT !.usable["en"]=171], [Good EXCEPT !.hits=Uniform(38)],
             [Good EXCEPT !.hits["en"]=[k \in Kinds |-> 45]],
             [Good EXCEPT !.usable["ko"]=193]}
Boundary == Gate(Good) /\ \A m \in Examples \ {Good} : ~Gate(m)
=============================================================================
