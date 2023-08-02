# @define
# class BaseGenome(ABC):
#     @abstractmethod
#     def get_sequence(self, chr: int, start: int, end: int) -> str:
#         raise NotImplementedError

#     # @abstractmethod
#     # def get_chr_len(self, chr: int) -> int:
#     #     # returns the length of the queried chromosome
#     #     raise NotImplementedError

#     @abstractmethod
#     def get_gene_sequence(self, gene: str) -> str:
#         raise NotImplementedError

#     # @abstractmethod
#     # def get_gene_position(self, gene: str) -> int:

#     #     raise NotImplementedError
#     @property
#     @abstractmethod
#     def translation_table(self) -> pd.DataFrame:
#         raise NotImplementedError

#     # @abstractmethod
#     # __getitem__(self, key: str) -> str:
