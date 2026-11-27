/* Private names for the shared libraries of a cnlp store (`cnlp-solve --privatize`).
 *
 * A library bundle carries its own copies of common libraries (OpenSSL, curl, libstdc++,
 * libblastrampoline, ...). The dynamic loader matches libraries by name, so a copy already
 * loaded under the same name is handed to whoever asks next: the host process's own (Python's
 * hashlib loads the system OpenSSL), or another runtime's (a model library and a solver each
 * carry a Julia runtime, and sharing one libblastrampoline lets one reset the other's BLAS).
 * `cnlp install` therefore gives each runtime's libraries names of their own: in every ELF
 * library of the store, each name that belongs to a library the store carries (DT_SONAME,
 * DT_NEEDED, the file names of its version needs) changes from `lib...` to a three-character
 * prefix of the directory that provides it (`c4k...`), the same length, so the string table is
 * edited in place (Julia's own LLVM and libjulia keep their names). So is the same name where
 * a runtime keeps it as a string to open by name (a NUL-terminated name standing alone), so a
 * runtime opens its own copy, not another runtime's of the same name. A link under the new name is
 * added beside every entry with the old one, and
 * the files keep their names, so whatever opens them by path still finds them. Running it
 * again changes nothing. ELF64 little-endian (Linux x86_64, aarch64); other files are left
 * alone.
 *
 * Scopes: each directory of $CNLPLIB_HOME/artifacts (a shared runtime), and each installed
 * library's or solver's directory (its lib/, the links its lib/julia makes into a runtime, and
 * the artifacts it carries in share/).
 * A name is renamed after the directory its file actually lives in, so a library and the
 * runtime it links agree. Two directories never share a prefix: the prefix chosen for each is
 * kept in $CNLPLIB_HOME/.privatize (prefix, then directory under the store, per line), and a
 * new directory whose first choice is taken gets the next free one, so a prefix, once given,
 * stays with its directory across installs.
 */
#include <fcntl.h>
#include <ftw.h>
#include <libgen.h>
#include <sys/mman.h>
#include <unistd.h>

typedef struct { char** v; int n, cap; } pv_list;
static void pv_push(pv_list* l, const char* s) {
    if (l->n == l->cap) { l->cap = l->cap ? 2 * l->cap : 64; l->v = realloc(l->v, l->cap * sizeof *l->v); }
    l->v[l->n++] = strdup(s);
}

/* the scope being processed: its ELF files, and its library names with the prefix of the
 * directory that provides each */
static pv_list PV_FILES, PV_NAMES, PV_PREFIX, PV_ENTRIES;
static int PV_LINKS;

/* the prefixes given so far, and the directory (under the store) each belongs to */
static pv_list PV_TAB_PRE, PV_TAB_DIR;
static void pv_table_path(char* out, size_t n) { snprintf(out, n, "%s/.privatize", home_dir()); }
static void pv_table_load(void) {
    char path[PATH_MAX]; pv_table_path(path, sizeof path);
    FILE* f = fopen(path, "r"); if (!f) return;
    char line[PATH_MAX + 8];
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\n")] = 0;
        char* tab = strchr(line, '\t');
        if (!tab || tab - line != 3) continue;
        *tab = 0; pv_push(&PV_TAB_PRE, line); pv_push(&PV_TAB_DIR, tab + 1);
    }
    fclose(f);
}
static void pv_table_save(void) {
    char path[PATH_MAX], tmp[PATH_MAX + 16]; pv_table_path(path, sizeof path);
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE* f = fopen(tmp, "w"); if (!f) die("cannot write %s", tmp);
    for (int k = 0; k < PV_TAB_PRE.n; k++) fprintf(f, "%s\t%s\n", PV_TAB_PRE.v[k], PV_TAB_DIR.v[k]);
    if (fclose(f) || rename(tmp, path)) die("cannot write %s", path);
}
/* three characters for a directory: the one it was given before, else the first free one of a
 * sequence seeded by its path under the store (stable if the store moves) */
static void pv_prefix(const char* dir, char out[4]) {
    const char* h = home_dir(); size_t L = strlen(h);
    const char* rel = !strncmp(dir, h, L) && dir[L] == '/' ? dir + L + 1 : dir;
    for (int k = 0; k < PV_TAB_DIR.n; k++)
        if (!strcmp(PV_TAB_DIR.v[k], rel)) { memcpy(out, PV_TAB_PRE.v[k], 4); return; }
    uint64_t x = 1469598103934665603ULL;
    for (const char* c = rel; *c; c++) { x ^= (unsigned char)*c; x *= 1099511628211ULL; }
    static const char A[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    for (int i = 0; i < 36 * 36; i++) {
        uint64_t y = (x + (uint64_t)i) % (36 * 36);
        out[0] = 'c'; out[1] = A[y % 36]; out[2] = A[y / 36]; out[3] = 0;
        int taken = 0;
        for (int k = 0; k < PV_TAB_PRE.n && !taken; k++) taken = !strcmp(PV_TAB_PRE.v[k], out);
        if (!taken) { pv_push(&PV_TAB_PRE, out); pv_push(&PV_TAB_DIR, rel); return; }
    }
    die("no private prefix left for %s (the store has %d library directories)", rel, PV_TAB_PRE.n);
}
static const char* pv_lookup(const char* name) {
    for (int k = 0; k < PV_NAMES.n; k++) if (!strcmp(PV_NAMES.v[k], name)) return PV_PREFIX.v[k];
    return NULL;
}

static int pv_is_elf(const char* p) {
    unsigned char h[6]; int fd = open(p, O_RDONLY); if (fd < 0) return 0;
    int ok = read(fd, h, 6) == 6 && !memcmp(h, "\177ELF", 4) && h[4] == 2 && h[5] == 1;   /* ELF64 LE */
    close(fd); return ok;
}
/* every `lib*.so*` entry of the scope (file or link) with the prefix of the directory its file
 * is in; every ELF file */
static int pv_scan(const char* p, const struct stat* st, int type, struct FTW* f) {
    (void)st;
    const char* base = p + f->base;
    /* Julia's own LLVM and libjulia keep their names: no host carries them (`libLLVM.so.18.1jl`
     * is Julia's build), and code loaded from outside the store may ask for them by name */
    int julia_own = !strncmp(base, "libLLVM", 7) || !strncmp(base, "libjulia", 8);
    /* nor does the GPU driver: libcuda (a bundle's copy is CUDA's forward-compatibility driver,
     * usable only on some machines) and libnvidia-* belong to the system's driver, which a
     * runtime must open by its standard name */
    julia_own |= !strncmp(base, "libcuda.", 8) || !strncmp(base, "libnvidia-", 10);
    if ((type == FTW_F || type == FTW_SL) && !strncmp(base, "lib", 3) && strstr(base, ".so") && !julia_own) {
        char real[PATH_MAX];
        if (realpath(p, real) && !pv_lookup(base)) {
            char pre[4]; pv_prefix(dirname(real), pre);
            pv_push(&PV_NAMES, base); pv_push(&PV_PREFIX, pre); pv_push(&PV_ENTRIES, p);
        }
    }
    if (type == FTW_F && strstr(base, ".so") && pv_is_elf(p)) pv_push(&PV_FILES, p);
    return 0;
}

/* The file offsets of the strings a mapped ELF64 file names libraries by: DT_SONAME,
 * DT_NEEDED, and its version needs' file names. */
typedef struct { uint32_t type, flags; uint64_t off, vaddr, paddr, filesz, memsz, align; } pv_phdr;
static int64_t pv_off(const unsigned char* b, uint64_t phoff, int phnum, uint64_t vaddr) {
    for (int k = 0; k < phnum; k++) {
        const pv_phdr* h = (const pv_phdr*)(b + phoff + (uint64_t)k * sizeof(pv_phdr));
        if (h->type == 1 && h->vaddr <= vaddr && vaddr < h->vaddr + h->filesz) return (int64_t)(h->off + vaddr - h->vaddr);
    }
    return -1;
}
static int pv_strings(const unsigned char* b, size_t size, uint64_t* out, int cap) {
    uint64_t phoff; uint16_t phnum; memcpy(&phoff, b + 0x20, 8); memcpy(&phnum, b + 0x38, 2);
    if (phoff + (uint64_t)phnum * sizeof(pv_phdr) > size) return 0;
    const pv_phdr* dyn = NULL;
    for (int k = 0; k < phnum; k++) { const pv_phdr* h = (const pv_phdr*)(b + phoff + (uint64_t)k * sizeof(pv_phdr)); if (h->type == 2) dyn = h; }
    if (!dyn || dyn->off + dyn->filesz > size) return 0;
    uint64_t strtab = 0, verneed = 0, nverneed = 0, refs[256]; int nref = 0;
    for (uint64_t o = dyn->off; o + 16 <= dyn->off + dyn->filesz; o += 16) {
        int64_t tag; uint64_t val; memcpy(&tag, b + o, 8); memcpy(&val, b + o + 8, 8);
        if (tag == 0) break;
        if (tag == 5) strtab = val;
        else if ((tag == 1 || tag == 14) && nref < 256) refs[nref++] = val;
        else if (tag == 0x6FFFFFFE) verneed = val;
        else if (tag == 0x6FFFFFFF) nverneed = val;
    }
    if (!strtab) return 0;
    int64_t st = pv_off(b, phoff, phnum, strtab); if (st < 0) return 0;
    if (verneed) {   /* Elf64_Verneed: vn_version, vn_cnt (16 bits), vn_file, vn_aux, vn_next (32 bits) */
        int64_t o = pv_off(b, phoff, phnum, verneed);
        for (uint64_t k = 0; o >= 0 && k < nverneed && nref < 256 && (uint64_t)o + 16 <= size; k++) {
            uint32_t file, next; memcpy(&file, b + o + 4, 4); memcpy(&next, b + o + 12, 4);
            refs[nref++] = file;
            if (!next) break;
            o += next;
        }
    }
    int n = 0;
    for (int k = 0; k < nref && n < cap; k++) {
        uint64_t s = (uint64_t)st + refs[k];
        int seen = 0; for (int j = 0; j < n; j++) seen |= out[j] == s;
        if (s < size && !seen) out[n++] = s;
    }
    return n;
}

/* rename, in one file, every library name of the scope; 1 if the file changed */
static int pv_edit(const char* p) {
    int fd = open(p, O_RDONLY); if (fd < 0) return 0;
    struct stat st; if (fstat(fd, &st) || st.st_size < 64) { close(fd); return 0; }
    size_t size = st.st_size;
    void* m = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    if (m == MAP_FAILED) return 0;
    const unsigned char* b = m;
    uint64_t offs[256]; int n = pv_strings(b, size, offs, 256), edits = 0, cap = 256;
    uint64_t* at = malloc(cap * sizeof *at); const char** to = malloc(cap * sizeof *to);
    if (!at || !to) die("out of memory");
    for (int k = 0; k < n; k++) {
        const char* s = (const char*)b + offs[k];
        const char* pre = strnlen(s, size - offs[k]) < size - offs[k] ? pv_lookup(s) : NULL;
        if (pre) { at[edits] = offs[k]; to[edits++] = pre; }
    }
    /* A library a runtime opens by name (`dlopen("libblastrampoline.so.5")`, a JLL's product
     * name) is a string in its data, not a DT_NEEDED entry: such a name, NUL-terminated and
     * standing alone (a NUL before it, not a path), is renamed too, so the runtime opens its
     * own private copy rather than one another runtime already loaded under that name. */
    for (const unsigned char* q = b + 1; (q = memmem(q, size - (q - b), "lib", 3)) != NULL; q += 3) {
        if (q[-1] != 0) continue;
        size_t left = size - (q - b), L = strnlen((const char*)q, left < 256 ? left : 256);
        if (L >= (left < 256 ? left : 256)) continue;
        const char* pre = pv_lookup((const char*)q);
        if (!pre) continue;
        uint64_t o = (uint64_t)(q - b); int seen = 0;
        for (int k = 0; k < edits && !seen; k++) seen = at[k] == o;
        if (seen) continue;
        if (edits == cap) {
            cap *= 2; at = realloc(at, cap * sizeof *at); to = realloc(to, cap * sizeof *to);
            if (!at || !to) die("out of memory");
        }
        at[edits] = o; to[edits++] = pre;
    }
    if (!edits) { free(at); free(to); munmap(m, size); return 0; }
    /* written to a new file and renamed over the old: a process that has it open keeps its copy */
    unsigned char* c = malloc(size);
    if (!c) { munmap(m, size); die("out of memory"); }
    memcpy(c, b, size); munmap(m, size);
    for (int k = 0; k < edits; k++) memcpy(c + at[k], to[k], 3);
    free(at); free(to);
    char tmp[PATH_MAX]; snprintf(tmp, sizeof tmp, "%s.cnlp-priv", p);
    FILE* f = fopen(tmp, "wb");
    if (!f) { free(c); die("cannot write %s", tmp); }
    size_t w = fwrite(c, 1, size, f); free(c);
    if (fclose(f) || w != size) die("cannot write %s", tmp);
    chmod(tmp, st.st_mode & 07777);
    if (rename(tmp, p)) die("cannot replace %s", p);
    return 1;
}

static void pv_clear(pv_list* l) { for (int k = 0; k < l->n; k++) free(l->v[k]); l->n = 0; }

/* one scope: its names (and the prefixes of their providers) recorded, its files edited */
static pv_list PV_ALL_ENTRIES, PV_ALL_PREFIXES;
static int pv_scope(const char* dir) {
    pv_clear(&PV_FILES); pv_clear(&PV_NAMES); pv_clear(&PV_PREFIX); pv_clear(&PV_ENTRIES);
    if (!isdir(dir)) return 0;
    nftw(dir, pv_scan, 32, FTW_PHYS);
    int changed = 0;
    for (int k = 0; k < PV_FILES.n; k++) changed += pv_edit(PV_FILES.v[k]);
    for (int k = 0; k < PV_ENTRIES.n; k++) {
        pv_push(&PV_ALL_ENTRIES, PV_ENTRIES.v[k]);
        int seen = 0;
        for (int j = 0; j < PV_ALL_PREFIXES.n; j++) seen |= !strcmp(PV_ALL_PREFIXES.v[j], PV_PREFIX.v[k]);
        if (!seen) pv_push(&PV_ALL_PREFIXES, PV_PREFIX.v[k]);
    }
    return changed;
}

/* A link under every private name beside every entry with the old one, to that entry. The
 * loader looks a needed name up in the directory of the link a library was loaded through
 * ($ORIGIN), and a library may carry its own copy of a runtime library (a newer libstdc++), so
 * whichever private name asks for it there finds the copy that directory chose. */
static void pv_links(void) {
    for (int k = 0; k < PV_ALL_ENTRIES.n; k++) {
        const char* e = PV_ALL_ENTRIES.v[k]; const char* base = strrchr(e, '/') + 1;
        for (int j = 0; j < PV_ALL_PREFIXES.n; j++) {
            char link[PATH_MAX]; snprintf(link, sizeof link, "%.*s%s%s", (int)(base - e), e, PV_ALL_PREFIXES.v[j], base + 3);
            struct stat ls;
            if (lstat(link, &ls) && !symlink(base, link)) PV_LINKS++;
        }
    }
}

static void pv_subdirs(pv_list* dirs, const char* kind, const char* sub) {
    char d[PATH_MAX]; snprintf(d, sizeof d, "%s/%s", home_dir(), kind);
    DIR* dd = opendir(d); if (!dd) return;
    for (struct dirent* e; (e = readdir(dd));) {
        if (e->d_name[0] == '.') continue;
        char p[PATH_MAX]; snprintf(p, sizeof p, "%s/%s%s", d, e->d_name, sub);
        if (isdir(p)) pv_push(dirs, p);
    }
    closedir(dd);
}

/* `cnlp-solve --privatize`: every shared runtime of the store, and every library and solver */
static int privatize(int nargs, char** args) {
    (void)args;
    if (nargs) die("--privatize takes no arguments: it works on the whole store ($CNLPLIB_HOME)");
    pv_table_load();
    pv_list dirs = { 0 };
    /* a library's or solver's whole directory: its lib/, and the artifacts its runtime loads
     * (share/julia/artifacts), in one scope, so their names agree */
    pv_subdirs(&dirs, "artifacts", ""); pv_subdirs(&dirs, "lib", ""); pv_subdirs(&dirs, "solvers", "");
    int changed = 0;
    for (int k = 0; k < dirs.n; k++) changed += pv_scope(dirs.v[k]);
    pv_links();
    pv_table_save();
    if (changed || PV_LINKS) printf("Gave the bundled libraries private names (%d files, %d links)\n", changed, PV_LINKS);
    return 0;
}
