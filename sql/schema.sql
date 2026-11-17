-- Tables served by the API (module 08)

create table if not exists predictions (
    run_id     text      not null,
    tx_id      bigint    not null,
    time_step  smallint  not null,
    score      real      not null,
    predicted  smallint  not null check (predicted in (0, 1)),
    label      smallint  check (label in (0, 1)),
    primary key (run_id, tx_id)
);

create index if not exists predictions_run_step_score
    on predictions (run_id, time_step, score desc);

create table if not exists metrics_by_time_step (
    run_id             text      not null,
    time_step          smallint  not null,
    n_illicit          integer   not null,
    precision_illicit  real      not null,
    recall_illicit     real      not null,
    f1_illicit         real      not null,
    primary key (run_id, time_step)
);
