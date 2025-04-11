CODEQL_DB="./codeql-dbs/torch/"
QUERY_DIR="./codeql-custom-queries-python/"
codeql database run-queries -- $CODEQL_DB "$QUERY_DIR/example.ql"
