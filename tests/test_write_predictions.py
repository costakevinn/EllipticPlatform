from write_predictions import connect, remove_stale


def test_remove_stale_deletes_only_runs_missing_from_mlflow():
    with connect() as conn, conn.transaction(force_rollback=True):
        rows = conn.execute("SELECT DISTINCT run_id FROM predictions").fetchall()
        alive = [row[0] for row in rows] + ["test-keep"]
        for run_id in ("test-keep", "test-stale"):
            conn.execute(
                "INSERT INTO predictions (run_id, tx_id, time_step, score, predicted) "
                "VALUES (%s, 1, 35, 0.5, 1)",
                (run_id,),
            )
            conn.execute(
                "INSERT INTO alerts (run_id, tx_id, time_step, score) "
                "VALUES (%s, 1, 35, 0.5)",
                (run_id,),
            )
        removed = remove_stale(conn, alive)
        assert removed["predictions"] == 1
        assert removed["alerts"] == 1
        left = conn.execute(
            "SELECT run_id FROM alerts WHERE run_id LIKE 'test-%'"
        ).fetchall()
        assert left == [("test-keep",)]
