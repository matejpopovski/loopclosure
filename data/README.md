# Intel Research Lab log

`intel.script.gz` is the robot log from the `intel_lab` record in Radish, as hosted
by DSpace@MIT.

- Record: <http://hdl.handle.net/1721.1/62287>
- Author on the record: Dieter Fox
- Licence on the record: CC0 1.0 Universal
- File MD5, as published on the record: `a12279da144d9a974520dffbcaf86693`
- `scripts/fetch_intel.sh` downloads it again and verifies that checksum.

The file is unchanged from the record. It holds 13,631 laser scans of 180 beams at
1 degree, in centimetres, with 8183 as the no-return code, and 26,915 odometry
poses in centimetres and degrees. Its timestamps start on 2000-12-05 UTC.

The record has no prose description. It was identified as the Intel Research Lab
log from its title, its author and the map image stored with it, which shows the
same building this project reconstructs.

The record also holds `rec2script.c`, the format converter, which is not copied
here. Its source was read to confirm the field layout. Nothing in this repository
is derived from it.
