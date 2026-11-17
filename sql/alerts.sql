create table if not exists alerts (
    run_id text not null,
    tx_id bigint not null,
    time_step smallint not null,
    score real not null,
    scored_at timestamptz not null default now(),
    primary key (run_id, tx_id)
);

create index if not exists alerts_run_step_score
    on alerts (run_id, time_step, score desc);
