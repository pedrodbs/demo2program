from ply import yacc
from .dsl_base import KarelDSLBase


class KarelDSLSyntax(KarelDSLBase):
    def get_yacc(self):
        self.yacc = yacc.yacc(
            method='LALR',
            debug=True,
            module=self,
            tabmodule='parsetab_syntax',
            # with_grammar=True
        )

        self.get_grammar()

    def get_next_candidates(self, code, **kwargs):
        next_candidates = self.yacc.parse(code, **kwargs)
        return next_candidates
