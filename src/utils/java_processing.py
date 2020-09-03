import javalang
import re

def check_and_fix_code_validity(train_df):
    valid_inds = []
    for i, s_code in enumerate(train_df.code):
        try:
            javalang.parse.parse_member_signature(s_code)
            valid_inds.append(True)
        except javalang.parser.JavaSyntaxError:
            try:
                modified_s_code = s_code + '\n}'
                javalang.parse.parse_member_signature(modified_s_code)
                valid_inds.append(True)
                train_df.code[i] = modified_s_code
            except javalang.parser.JavaSyntaxError:
                valid_inds.append(False)
    return valid_inds

def split_java_token(cstr, camel_case=True, split_char='_'):
    res_split = []

    if camel_case:
        res_split = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', cstr)).split()

    if split_char:
        char_splt = []
        if not camel_case:
            res_split = [cstr]
        for token in res_split:
            char_splt += token.split(split_char)
        res_split = char_splt
    return [token for token in res_split if len(token) > 0]

def tokenize_java_code(cstr):
    stop_tokens = {'{', '}', ';'}
    return [token for plain_token in javalang.tokenizer.tokenize(cstr) \
            if not plain_token.value in stop_tokens \
            for token in split_java_token(plain_token.value, camel_case=True, split_char='_')]


def get_api_sequence(cstr, split_api_tokens=False):
    def find_method(method_node, filter):
        sub_api_seq = []
        for node in method_node.arguments:
            if isinstance(node, javalang.tree.MethodInvocation):
                api = [filter.get(node.qualifier, node.qualifier),
                       node.member]
                sub_api_seq.append(api)
                sub_api_seq.extend(find_method(node, filter))

            if isinstance(node, javalang.tree.ClassCreator):
                api = [get_last_sub_type(node.type).name, 'new']
                sub_api_seq.append(api)
                sub_api_seq.extend(find_method(node, filter))
        return sub_api_seq

    def check_selectors(node, s_filter):
        select_api_seq = []
        if node.selectors is not None:
            for sel in node.selectors:
                if isinstance(sel, javalang.tree.MethodInvocation):
                    if node.qualifier is None:
                        select_api_seq.append([get_last_sub_type(node.type).name, sel.member])
                    else:
                        select_api_seq.append(
                            [s_filter.get(node.qualifier, node.qualifier),
                             sel.member])
        return select_api_seq

    def get_last_sub_type(node):
        if not 'sub_type' in node.attrs or not node.sub_type:
            return node
        else:
            return get_last_sub_type(node.sub_type)

    api_seq = []
    tree = javalang.parse.parse_member_signature(cstr)
    identifier_filter = {}
    this_selectors = []
    for _, node in tree:
        if isinstance(node, javalang.tree.FormalParameter):
            identifier_filter[node.name] = get_last_sub_type(node.type).name

        if isinstance(node, javalang.tree.LocalVariableDeclaration):
            for dec in node.declarators:
                identifier_filter[dec.name] = get_last_sub_type(node.type).name

        if isinstance(node, javalang.tree.ClassCreator):
            api = [get_last_sub_type(node.type).name, 'new']
            api_seq.append(api)
            api_seq.extend(check_selectors(node, identifier_filter))

        if isinstance(node, javalang.tree.MethodInvocation):

            if node.qualifier is None:
                if len(api_seq) != 0:
                    node.qualifier = api_seq[-1][0]
                elif len(this_selectors) != 0:
                    try:
                        node_pos = this_selectors.index(node)
                        if isinstance(this_selectors[node_pos - 1], javalang.tree.MemberReference):
                            node.qualifier = this_selectors[node_pos - 1].member
                    except ValueError:
                        node.qualifier = ''

            sub_api_seq = find_method(node, identifier_filter)
            sub_api_seq.append(
                [identifier_filter.get(node.qualifier, node.qualifier),
                 node.member])
            api_seq.extend(sub_api_seq)
            api_seq.extend(check_selectors(node, identifier_filter))

        if isinstance(node, javalang.tree.This):
            this_selectors = node.selectors
    api_seq = [item for pairs in api_seq for item in pairs if item]
    if split_api_tokens:
        api_seq = [token for item in api_seq for token in split_java_token(item)]
    return api_seq