replace "pysol_update_array(src, src_v, dst, dst_v)":

$$import _read_ghost_$$type($src)(in int n, in int m, in int ghost_edge_len, in Array<$$type($src)> a[n][m], out Array<$$type($src)> ghost_edge[ghost_edge_len][1]);
$$import _write_ghost_$$type($src)(in int n, in int m, in int ghost_edge_len, in Array<$$type($src)> left[ghost_edge_len][1], in Array<$$type($src)> right[ghost_edge_len][1], in Array<$$type($src)> up[ghost_edge_len][1], in Array<$$type($src)> down[ghost_edge_len][1], in Array<$$type($src)> A[n][m], out Array<$$type($src)> B[n][m]);

$$declare: int _Update_SRC_block_n1 = %block_n($src,  0)-2;
$$declare: int _Update_SRC_block_n2 = %block_n($src,  1)-2;

$$declare: int _Update_SRC_n1_blocks = %n_blocks($src,  0);
$$declare: int _Update_SRC_n2_blocks = %n_blocks($src,  1);

$$declare: int _ghost_edges_len = _Update_SRC_block_n1 * 2 + _Update_SRC_block_n2 * 2;
$$declare: Array<$$type($src)[_ghost_edges_len][1]> ghost_edges[_Update_SRC_n1_blocks][_Update_SRC_n2_blocks];

for (int i = 0; i < _Update_SRC_n1_blocks; i++)
{
	for (int j = 0; j < _Update_SRC_n2_blocks; ++j)
	{
		_read_ghost_$$type($src)(_Update_SRC_block_n1, _Update_SRC_block_n2, _ghost_edges_len, $src{$src_v}[i][j], ghost_edges[i][j]);
	}
}

	_write_ghost_$$type($src)(
		_Update_SRC_block_n1,
		_Update_SRC_block_n2,
		_ghost_edges_len,
		ghost_edges[0][0],
		ghost_edges[0][1],
		ghost_edges[0][0],
		ghost_edges[1][0],
		$src{$src_v}[0][0],
		$dst{$dst_v}[0][0]);

	_write_ghost_$$type($src)(
		_Update_SRC_block_n1,
		_Update_SRC_block_n2,
		_ghost_edges_len,
		ghost_edges[0][_Update_SRC_n2_blocks-2],
		ghost_edges[0][_Update_SRC_n2_blocks-1],
		ghost_edges[0][_Update_SRC_n2_blocks-1],
		ghost_edges[1][_Update_SRC_n2_blocks-1],
		$src{$src_v}[0][_Update_SRC_n2_blocks-1],
		$dst{$dst_v}[0][_Update_SRC_n2_blocks-1]);

	_write_ghost_$$type($src)(
		_Update_SRC_block_n1,
		_Update_SRC_block_n2,
		_ghost_edges_len,
		ghost_edges[_Update_SRC_n1_blocks-1][0],
		ghost_edges[_Update_SRC_n1_blocks-1][1],
		ghost_edges[_Update_SRC_n1_blocks-2][0],
		ghost_edges[_Update_SRC_n1_blocks-1][0],
		$src{$src_v}[_Update_SRC_n1_blocks-1][0],
		$dst{$dst_v}[_Update_SRC_n1_blocks-1][0]);

	_write_ghost_$$type($src)(
		_Update_SRC_block_n1,
		_Update_SRC_block_n2,
		_ghost_edges_len,
		ghost_edges[_Update_SRC_n1_blocks-1][_Update_SRC_n2_blocks-2],
		ghost_edges[_Update_SRC_n1_blocks-1][_Update_SRC_n2_blocks-1],
		ghost_edges[_Update_SRC_n1_blocks-2][_Update_SRC_n2_blocks-1],
		ghost_edges[_Update_SRC_n1_blocks-1][_Update_SRC_n2_blocks-1],
		$src{$src_v}[_Update_SRC_n1_blocks-1][_Update_SRC_n2_blocks-1],
		$dst{$dst_v}[_Update_SRC_n1_blocks-1][_Update_SRC_n2_blocks-1]);

	// Left
	for (int i = 1; i < _Update_SRC_n1_blocks - 1; i++)
	{
		_write_ghost_$$type($src)(
			_Update_SRC_block_n1,
			_Update_SRC_block_n2,
			_ghost_edges_len,
			ghost_edges[i][0],
			ghost_edges[i][1],
			ghost_edges[i-1][0],
			ghost_edges[i+1][0],
			$src{$src_v}[i][0],
			$dst{$dst_v}[i][0]);
	}
	// Right
	for (int i = 1; i < _Update_SRC_n1_blocks - 1; i++)
	{
		_write_ghost_$$type($src)(
			_Update_SRC_block_n1,
			_Update_SRC_block_n2,
			_ghost_edges_len,
			ghost_edges[i][_Update_SRC_n2_blocks-2],
			ghost_edges[i][_Update_SRC_n2_blocks-1],
			ghost_edges[i-1][_Update_SRC_n2_blocks-1],
			ghost_edges[i+1][_Update_SRC_n2_blocks-1],
			$src{$src_v}[i][_Update_SRC_n2_blocks-1],
			$dst{$dst_v}[i][_Update_SRC_n2_blocks-1]);
	}
	// Up
	for (int j = 1; j < _Update_SRC_n2_blocks - 1; j++)
	{
		_write_ghost_$$type($src)(
			_Update_SRC_block_n1,
			_Update_SRC_block_n2,
			_ghost_edges_len,
			ghost_edges[0][j-1],
			ghost_edges[0][j+1],
			ghost_edges[0][j],
			ghost_edges[1][j],
			$src{$src_v}[0][j],
			$dst{$dst_v}[0][j]);
	}
	// Down
	for (int j = 1; j < _Update_SRC_n2_blocks - 1; j++)
	{
		_write_ghost_$$type($src)(
			_Update_SRC_block_n1,
			_Update_SRC_block_n2,
			_ghost_edges_len,
			ghost_edges[_Update_SRC_n1_blocks-1][j-1],
			ghost_edges[_Update_SRC_n1_blocks-1][j+1],
			ghost_edges[_Update_SRC_n1_blocks-2][j],
			ghost_edges[_Update_SRC_n1_blocks-1][j],
			$src{$src_v}[_Update_SRC_n1_blocks-1][j],
			$dst{$dst_v}[_Update_SRC_n1_blocks-1][j]);
	}
// Center
for (int i = 1; i < _Update_SRC_n1_blocks - 1; i++)
{
	for (int j = 1; j < _Update_SRC_n2_blocks -1; j++)
	{
		_write_ghost_$$type($src)(
			_Update_SRC_block_n1,
			_Update_SRC_block_n2,
			_ghost_edges_len,
			ghost_edges[i][j-1],
			ghost_edges[i][j+1],
			ghost_edges[i - 1][j],
			ghost_edges[i + 1][j],
			$src{$src_v}[i][j],
			$dst{$dst_v}[i][j]);
	}
}
