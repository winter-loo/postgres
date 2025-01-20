create table t1(a int);
insert into t1 values (1);

truncate table t1;
insert into t1 values (1);

delete from t1 where a = 1;
insert into t1 values (1);

select itime as myitime, ctid as myctid from t1 \gset

select ctid as myctid2 from t1 where itime = :myitime \gset

select :myctid = :myctid2;
--= true
