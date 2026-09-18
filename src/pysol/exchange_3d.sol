replace "pysol_update_array3d(src, src_v, dst, dst_v)":

$$import _read_ghost3d_$$type($src)(in int n, in int m, in int p, in int ghost_face_len, in Array<$$type($src)> a[n][m][p], out Array<$$type($src)> ghost_face[ghost_face_len][1]);
$$import _write_ghost3d_$$type($src)(in int n, in int m, in int p, in int ghost_face_len, in Array<$$type($src)> left[ghost_face_len][1], in Array<$$type($src)> right[ghost_face_len][1], in Array<$$type($src)> up[ghost_face_len][1], in Array<$$type($src)> down[ghost_face_len][1], in Array<$$type($src)> front[ghost_face_len][1], in Array<$$type($src)> back[ghost_face_len][1], in Array<$$type($src)> A[n][m][p], out Array<$$type($src)> B[n][m][p]);

// Axis convention (matches exchange.sol's left/right/up/down for 2D):
//   i (dim 0) is the vertical axis   — up (i=0)    / down  (i=n1-1)
//   j (dim 1) is the horizontal axis — left (j=0)  / right (j=n2-1)
//   k (dim 2) is the z-axis / depth  — front (k=0) / back  (k=n3-1)

$$declare: int _Update3D_SRC_block_n1 = %block_n($src, 0) - 2;
$$declare: int _Update3D_SRC_block_n2 = %block_n($src, 1) - 2;
$$declare: int _Update3D_SRC_block_n3 = %block_n($src, 2) - 2;

$$declare: int _Update3D_SRC_n1_blocks = %n_blocks($src, 0);
$$declare: int _Update3D_SRC_n2_blocks = %n_blocks($src, 1);
$$declare: int _Update3D_SRC_n3_blocks = %n_blocks($src, 2);

$$declare: int _ghost_faces_len = _Update3D_SRC_block_n1 * _Update3D_SRC_block_n2 * 2 + _Update3D_SRC_block_n1 * _Update3D_SRC_block_n3 * 2 + _Update3D_SRC_block_n2 * _Update3D_SRC_block_n3 * 2;
$$declare: Array<$$type($src)[_ghost_faces_len][1]> ghost_faces[_Update3D_SRC_n1_blocks][_Update3D_SRC_n2_blocks][_Update3D_SRC_n3_blocks];

for (int i = 0; i < _Update3D_SRC_n1_blocks; i++)
{
	for (int j = 0; j < _Update3D_SRC_n2_blocks; j++)
	{
		for (int k = 0; k < _Update3D_SRC_n3_blocks; k++)
		{
			_read_ghost3d_$$type($src)(_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len, $src{$src_v}[i][j][k], ghost_faces[i][j][k]);
		}
	}
}

// Corner (up, left, front): i=0, j=0, k=0
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[0][0][0],
		ghost_faces[0][1][0],
		ghost_faces[0][0][0],
		ghost_faces[1][0][0],
		ghost_faces[0][0][0],
		ghost_faces[0][0][1],
		$src{$src_v}[0][0][0],
		$dst{$dst_v}[0][0][0]);

// Corner (up, left, back): i=0, j=0, k=n3-1
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[0][0][_Update3D_SRC_n3_blocks-1],
		ghost_faces[0][1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[0][0][_Update3D_SRC_n3_blocks-1],
		ghost_faces[1][0][_Update3D_SRC_n3_blocks-1],
		ghost_faces[0][0][_Update3D_SRC_n3_blocks-2],
		ghost_faces[0][0][_Update3D_SRC_n3_blocks-1],
		$src{$src_v}[0][0][_Update3D_SRC_n3_blocks-1],
		$dst{$dst_v}[0][0][_Update3D_SRC_n3_blocks-1]);

// Corner (up, right, front): i=0, j=n2-1, k=0
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[0][_Update3D_SRC_n2_blocks-2][0],
		ghost_faces[0][_Update3D_SRC_n2_blocks-1][0],
		ghost_faces[0][_Update3D_SRC_n2_blocks-1][0],
		ghost_faces[1][_Update3D_SRC_n2_blocks-1][0],
		ghost_faces[0][_Update3D_SRC_n2_blocks-1][0],
		ghost_faces[0][_Update3D_SRC_n2_blocks-1][1],
		$src{$src_v}[0][_Update3D_SRC_n2_blocks-1][0],
		$dst{$dst_v}[0][_Update3D_SRC_n2_blocks-1][0]);

// Corner (up, right, back): i=0, j=n2-1, k=n3-1
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[0][_Update3D_SRC_n2_blocks-2][_Update3D_SRC_n3_blocks-1],
		ghost_faces[0][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[0][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[1][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[0][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-2],
		ghost_faces[0][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		$src{$src_v}[0][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		$dst{$dst_v}[0][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1]);

// Corner (down, left, front): i=n1-1, j=0, k=0
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[_Update3D_SRC_n1_blocks-1][0][0],
		ghost_faces[_Update3D_SRC_n1_blocks-1][1][0],
		ghost_faces[_Update3D_SRC_n1_blocks-2][0][0],
		ghost_faces[_Update3D_SRC_n1_blocks-1][0][0],
		ghost_faces[_Update3D_SRC_n1_blocks-1][0][0],
		ghost_faces[_Update3D_SRC_n1_blocks-1][0][1],
		$src{$src_v}[_Update3D_SRC_n1_blocks-1][0][0],
		$dst{$dst_v}[_Update3D_SRC_n1_blocks-1][0][0]);

// Corner (down, left, back): i=n1-1, j=0, k=n3-1
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[_Update3D_SRC_n1_blocks-1][0][_Update3D_SRC_n3_blocks-1],
		ghost_faces[_Update3D_SRC_n1_blocks-1][1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[_Update3D_SRC_n1_blocks-2][0][_Update3D_SRC_n3_blocks-1],
		ghost_faces[_Update3D_SRC_n1_blocks-1][0][_Update3D_SRC_n3_blocks-1],
		ghost_faces[_Update3D_SRC_n1_blocks-1][0][_Update3D_SRC_n3_blocks-2],
		ghost_faces[_Update3D_SRC_n1_blocks-1][0][_Update3D_SRC_n3_blocks-1],
		$src{$src_v}[_Update3D_SRC_n1_blocks-1][0][_Update3D_SRC_n3_blocks-1],
		$dst{$dst_v}[_Update3D_SRC_n1_blocks-1][0][_Update3D_SRC_n3_blocks-1]);

// Corner (down, right, front): i=n1-1, j=n2-1, k=0
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-2][0],
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][0],
		ghost_faces[_Update3D_SRC_n1_blocks-2][_Update3D_SRC_n2_blocks-1][0],
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][0],
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][0],
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][1],
		$src{$src_v}[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][0],
		$dst{$dst_v}[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][0]);

// Corner (down, right, back): i=n1-1, j=n2-1, k=n3-1
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-2][_Update3D_SRC_n3_blocks-1],
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[_Update3D_SRC_n1_blocks-2][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-2],
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		$src{$src_v}[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		$dst{$dst_v}[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1]);

// Edge along vertical (i), at left-front: j=0, k=0
for (int i = 1; i < _Update3D_SRC_n1_blocks - 1; i++)
{
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[i][0][0],
		ghost_faces[i][1][0],
		ghost_faces[i-1][0][0],
		ghost_faces[i+1][0][0],
		ghost_faces[i][0][0],
		ghost_faces[i][0][1],
		$src{$src_v}[i][0][0],
		$dst{$dst_v}[i][0][0]);
}
// Edge along vertical (i), at left-back: j=0, k=n3-1
for (int i = 1; i < _Update3D_SRC_n1_blocks - 1; i++)
{
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[i][0][_Update3D_SRC_n3_blocks-1],
		ghost_faces[i][1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[i-1][0][_Update3D_SRC_n3_blocks-1],
		ghost_faces[i+1][0][_Update3D_SRC_n3_blocks-1],
		ghost_faces[i][0][_Update3D_SRC_n3_blocks-2],
		ghost_faces[i][0][_Update3D_SRC_n3_blocks-1],
		$src{$src_v}[i][0][_Update3D_SRC_n3_blocks-1],
		$dst{$dst_v}[i][0][_Update3D_SRC_n3_blocks-1]);
}
// Edge along vertical (i), at right-front: j=n2-1, k=0
for (int i = 1; i < _Update3D_SRC_n1_blocks - 1; i++)
{
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[i][_Update3D_SRC_n2_blocks-2][0],
		ghost_faces[i][_Update3D_SRC_n2_blocks-1][0],
		ghost_faces[i-1][_Update3D_SRC_n2_blocks-1][0],
		ghost_faces[i+1][_Update3D_SRC_n2_blocks-1][0],
		ghost_faces[i][_Update3D_SRC_n2_blocks-1][0],
		ghost_faces[i][_Update3D_SRC_n2_blocks-1][1],
		$src{$src_v}[i][_Update3D_SRC_n2_blocks-1][0],
		$dst{$dst_v}[i][_Update3D_SRC_n2_blocks-1][0]);
}
// Edge along vertical (i), at right-back: j=n2-1, k=n3-1
for (int i = 1; i < _Update3D_SRC_n1_blocks - 1; i++)
{
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[i][_Update3D_SRC_n2_blocks-2][_Update3D_SRC_n3_blocks-1],
		ghost_faces[i][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[i-1][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[i+1][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[i][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-2],
		ghost_faces[i][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		$src{$src_v}[i][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1],
		$dst{$dst_v}[i][_Update3D_SRC_n2_blocks-1][_Update3D_SRC_n3_blocks-1]);
}
// Edge along horizontal (j), at up-front: i=0, k=0
for (int j = 1; j < _Update3D_SRC_n2_blocks - 1; j++)
{
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[0][j-1][0],
		ghost_faces[0][j+1][0],
		ghost_faces[0][j][0],
		ghost_faces[1][j][0],
		ghost_faces[0][j][0],
		ghost_faces[0][j][1],
		$src{$src_v}[0][j][0],
		$dst{$dst_v}[0][j][0]);
}
// Edge along horizontal (j), at up-back: i=0, k=n3-1
for (int j = 1; j < _Update3D_SRC_n2_blocks - 1; j++)
{
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[0][j-1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[0][j+1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[0][j][_Update3D_SRC_n3_blocks-1],
		ghost_faces[1][j][_Update3D_SRC_n3_blocks-1],
		ghost_faces[0][j][_Update3D_SRC_n3_blocks-2],
		ghost_faces[0][j][_Update3D_SRC_n3_blocks-1],
		$src{$src_v}[0][j][_Update3D_SRC_n3_blocks-1],
		$dst{$dst_v}[0][j][_Update3D_SRC_n3_blocks-1]);
}
// Edge along horizontal (j), at down-front: i=n1-1, k=0
for (int j = 1; j < _Update3D_SRC_n2_blocks - 1; j++)
{
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[_Update3D_SRC_n1_blocks-1][j-1][0],
		ghost_faces[_Update3D_SRC_n1_blocks-1][j+1][0],
		ghost_faces[_Update3D_SRC_n1_blocks-2][j][0],
		ghost_faces[_Update3D_SRC_n1_blocks-1][j][0],
		ghost_faces[_Update3D_SRC_n1_blocks-1][j][0],
		ghost_faces[_Update3D_SRC_n1_blocks-1][j][1],
		$src{$src_v}[_Update3D_SRC_n1_blocks-1][j][0],
		$dst{$dst_v}[_Update3D_SRC_n1_blocks-1][j][0]);
}
// Edge along horizontal (j), at down-back: i=n1-1, k=n3-1
for (int j = 1; j < _Update3D_SRC_n2_blocks - 1; j++)
{
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[_Update3D_SRC_n1_blocks-1][j-1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[_Update3D_SRC_n1_blocks-1][j+1][_Update3D_SRC_n3_blocks-1],
		ghost_faces[_Update3D_SRC_n1_blocks-2][j][_Update3D_SRC_n3_blocks-1],
		ghost_faces[_Update3D_SRC_n1_blocks-1][j][_Update3D_SRC_n3_blocks-1],
		ghost_faces[_Update3D_SRC_n1_blocks-1][j][_Update3D_SRC_n3_blocks-2],
		ghost_faces[_Update3D_SRC_n1_blocks-1][j][_Update3D_SRC_n3_blocks-1],
		$src{$src_v}[_Update3D_SRC_n1_blocks-1][j][_Update3D_SRC_n3_blocks-1],
		$dst{$dst_v}[_Update3D_SRC_n1_blocks-1][j][_Update3D_SRC_n3_blocks-1]);
}
// Edge along z (k), at up-left: i=0, j=0
for (int k = 1; k < _Update3D_SRC_n3_blocks - 1; k++)
{
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[0][0][k],
		ghost_faces[0][1][k],
		ghost_faces[0][0][k],
		ghost_faces[1][0][k],
		ghost_faces[0][0][k-1],
		ghost_faces[0][0][k+1],
		$src{$src_v}[0][0][k],
		$dst{$dst_v}[0][0][k]);
}
// Edge along z (k), at up-right: i=0, j=n2-1
for (int k = 1; k < _Update3D_SRC_n3_blocks - 1; k++)
{
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[0][_Update3D_SRC_n2_blocks-2][k],
		ghost_faces[0][_Update3D_SRC_n2_blocks-1][k],
		ghost_faces[0][_Update3D_SRC_n2_blocks-1][k],
		ghost_faces[1][_Update3D_SRC_n2_blocks-1][k],
		ghost_faces[0][_Update3D_SRC_n2_blocks-1][k-1],
		ghost_faces[0][_Update3D_SRC_n2_blocks-1][k+1],
		$src{$src_v}[0][_Update3D_SRC_n2_blocks-1][k],
		$dst{$dst_v}[0][_Update3D_SRC_n2_blocks-1][k]);
}
// Edge along z (k), at down-left: i=n1-1, j=0
for (int k = 1; k < _Update3D_SRC_n3_blocks - 1; k++)
{
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[_Update3D_SRC_n1_blocks-1][0][k],
		ghost_faces[_Update3D_SRC_n1_blocks-1][1][k],
		ghost_faces[_Update3D_SRC_n1_blocks-2][0][k],
		ghost_faces[_Update3D_SRC_n1_blocks-1][0][k],
		ghost_faces[_Update3D_SRC_n1_blocks-1][0][k-1],
		ghost_faces[_Update3D_SRC_n1_blocks-1][0][k+1],
		$src{$src_v}[_Update3D_SRC_n1_blocks-1][0][k],
		$dst{$dst_v}[_Update3D_SRC_n1_blocks-1][0][k]);
}
// Edge along z (k), at down-right: i=n1-1, j=n2-1
for (int k = 1; k < _Update3D_SRC_n3_blocks - 1; k++)
{
	_write_ghost3d_$$type($src)(
		_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-2][k],
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][k],
		ghost_faces[_Update3D_SRC_n1_blocks-2][_Update3D_SRC_n2_blocks-1][k],
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][k],
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][k-1],
		ghost_faces[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][k+1],
		$src{$src_v}[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][k],
		$dst{$dst_v}[_Update3D_SRC_n1_blocks-1][_Update3D_SRC_n2_blocks-1][k]);
}
// Face up (vertical, i=0)
for (int j = 1; j < _Update3D_SRC_n2_blocks - 1; j++)
{
	for (int k = 1; k < _Update3D_SRC_n3_blocks - 1; k++)
	{
		_write_ghost3d_$$type($src)(
			_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
			ghost_faces[0][j-1][k],
			ghost_faces[0][j+1][k],
			ghost_faces[0][j][k],
			ghost_faces[1][j][k],
			ghost_faces[0][j][k-1],
			ghost_faces[0][j][k+1],
			$src{$src_v}[0][j][k],
			$dst{$dst_v}[0][j][k]);
	}
}
// Face down (vertical, i=n1-1)
for (int j = 1; j < _Update3D_SRC_n2_blocks - 1; j++)
{
	for (int k = 1; k < _Update3D_SRC_n3_blocks - 1; k++)
	{
		_write_ghost3d_$$type($src)(
			_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
			ghost_faces[_Update3D_SRC_n1_blocks-1][j-1][k],
			ghost_faces[_Update3D_SRC_n1_blocks-1][j+1][k],
			ghost_faces[_Update3D_SRC_n1_blocks-2][j][k],
			ghost_faces[_Update3D_SRC_n1_blocks-1][j][k],
			ghost_faces[_Update3D_SRC_n1_blocks-1][j][k-1],
			ghost_faces[_Update3D_SRC_n1_blocks-1][j][k+1],
			$src{$src_v}[_Update3D_SRC_n1_blocks-1][j][k],
			$dst{$dst_v}[_Update3D_SRC_n1_blocks-1][j][k]);
	}
}
// Face left (horizontal, j=0)
for (int i = 1; i < _Update3D_SRC_n1_blocks - 1; i++)
{
	for (int k = 1; k < _Update3D_SRC_n3_blocks - 1; k++)
	{
		_write_ghost3d_$$type($src)(
			_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
			ghost_faces[i][0][k],
			ghost_faces[i][1][k],
			ghost_faces[i-1][0][k],
			ghost_faces[i+1][0][k],
			ghost_faces[i][0][k-1],
			ghost_faces[i][0][k+1],
			$src{$src_v}[i][0][k],
			$dst{$dst_v}[i][0][k]);
	}
}
// Face right (horizontal, j=n2-1)
for (int i = 1; i < _Update3D_SRC_n1_blocks - 1; i++)
{
	for (int k = 1; k < _Update3D_SRC_n3_blocks - 1; k++)
	{
		_write_ghost3d_$$type($src)(
			_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
			ghost_faces[i][_Update3D_SRC_n2_blocks-2][k],
			ghost_faces[i][_Update3D_SRC_n2_blocks-1][k],
			ghost_faces[i-1][_Update3D_SRC_n2_blocks-1][k],
			ghost_faces[i+1][_Update3D_SRC_n2_blocks-1][k],
			ghost_faces[i][_Update3D_SRC_n2_blocks-1][k-1],
			ghost_faces[i][_Update3D_SRC_n2_blocks-1][k+1],
			$src{$src_v}[i][_Update3D_SRC_n2_blocks-1][k],
			$dst{$dst_v}[i][_Update3D_SRC_n2_blocks-1][k]);
	}
}
// Face front (z, k=0)
for (int i = 1; i < _Update3D_SRC_n1_blocks - 1; i++)
{
	for (int j = 1; j < _Update3D_SRC_n2_blocks - 1; j++)
	{
		_write_ghost3d_$$type($src)(
			_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
			ghost_faces[i][j-1][0],
			ghost_faces[i][j+1][0],
			ghost_faces[i-1][j][0],
			ghost_faces[i+1][j][0],
			ghost_faces[i][j][0],
			ghost_faces[i][j][1],
			$src{$src_v}[i][j][0],
			$dst{$dst_v}[i][j][0]);
	}
}
// Face back (z, k=n3-1)
for (int i = 1; i < _Update3D_SRC_n1_blocks - 1; i++)
{
	for (int j = 1; j < _Update3D_SRC_n2_blocks - 1; j++)
	{
		_write_ghost3d_$$type($src)(
			_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
			ghost_faces[i][j-1][_Update3D_SRC_n3_blocks-1],
			ghost_faces[i][j+1][_Update3D_SRC_n3_blocks-1],
			ghost_faces[i-1][j][_Update3D_SRC_n3_blocks-1],
			ghost_faces[i+1][j][_Update3D_SRC_n3_blocks-1],
			ghost_faces[i][j][_Update3D_SRC_n3_blocks-2],
			ghost_faces[i][j][_Update3D_SRC_n3_blocks-1],
			$src{$src_v}[i][j][_Update3D_SRC_n3_blocks-1],
			$dst{$dst_v}[i][j][_Update3D_SRC_n3_blocks-1]);
	}
}
// Center
for (int i = 1; i < _Update3D_SRC_n1_blocks - 1; i++)
{
	for (int j = 1; j < _Update3D_SRC_n2_blocks - 1; j++)
	{
		for (int k = 1; k < _Update3D_SRC_n3_blocks - 1; k++)
		{
			_write_ghost3d_$$type($src)(
				_Update3D_SRC_block_n1, _Update3D_SRC_block_n2, _Update3D_SRC_block_n3, _ghost_faces_len,
				ghost_faces[i][j-1][k],
				ghost_faces[i][j+1][k],
				ghost_faces[i-1][j][k],
				ghost_faces[i+1][j][k],
				ghost_faces[i][j][k-1],
				ghost_faces[i][j][k+1],
				$src{$src_v}[i][j][k],
				$dst{$dst_v}[i][j][k]);
		}
	}
}
