def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim