import sys
import helmet

if len(sys.argv) < 3:
    exit(0)

helmet.generateHelm(sys.argv[1], sys.argv[2])
